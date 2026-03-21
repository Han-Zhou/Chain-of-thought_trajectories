#!/bin/bash
# ==============================================================================
# SLURM MONITOR - FINAL FIX (STABLE PROBE & FIXED FOOTER)
# ==============================================================================
# --- Configuration ---
STATE_DB="$HOME/.slurm_monitor_db.log"
RENDER_FILE="$HOME/.slurm_monitor_render.tmp"
GPU_FILE="$HOME/.slurm_monitor_gpu.tmp"
NODE_FILE="$HOME/.slurm_monitor_node.tmp"
SHARED_FILE="$HOME/.slurm_monitor_shared.tmp"
HOME_FILE="$HOME/.slurm_monitor_home.tmp"
LOCK_FILE="$HOME/.slurm_monitor.lock"
REFRESH_RATE=1
# --- Colors ---
R=$(tput setaf 1); G=$(tput setaf 2); Y=$(tput setaf 3)
B=$(tput setaf 4); C=$(tput setaf 6); W=$(tput setaf 7)
Dim=$(tput dim); Bold=$(tput bold); Reset=$(tput sgr0)
BgR=$(tput setab 1); BgG=$(tput setab 2); BgY=$'\033[48;5;136m'; BgW=$(tput setab 7); Blk=$(tput setaf 0)
ClearLine='\033[K'
# --- Cleanup ---
cleanup() {
    [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null
    tput rmcup; tput cnorm; stty echo; exit 0
}
trap cleanup SIGINT SIGTERM
# --- SETUP ---
tput smcup; tput civis; stty -echo
tput clear 
touch "$STATE_DB" "$RENDER_FILE" "$GPU_FILE" "$NODE_FILE" "$SHARED_FILE" "$HOME_FILE"
# ==============================================================================
# 1. HELPER: Time Parsing
# ==============================================================================
parse_slurm_to_seconds() {
    local input=$1; local days=0; local hms=""
    if [[ "$input" == *"-"* ]]; then days=${input%%-*}; hms=${input#*-}; else hms=$input; fi
    IFS=: read -r -a parts <<< "$hms"
    local seconds=0
    if [[ ${#parts[@]} -eq 3 ]]; then seconds=$(( 10#${parts[0]} * 3600 + 10#${parts[1]} * 60 + 10#${parts[2]} ))
    elif [[ ${#parts[@]} -eq 2 ]]; then seconds=$(( 10#${parts[0]} * 60 + 10#${parts[1]} ))
    else seconds=$(( 10#${parts[0]} )); fi
    echo $(( days * 86400 + seconds ))
}
# ==============================================================================
# 2. BACKEND PROCESS
# ==============================================================================
backend_loop() {
    while true; do
        (
            flock -x 200 || exit 1
            declare -A JOB_DB
            [[ -f "$STATE_DB" ]] && while read -r jid start_ts; do [[ -n "$jid" ]] && JOB_DB[$jid]=$start_ts; done < "$STATE_DB"
            # Fetch ALL data
            raw_data=$(squeue --all --noheader --sort=t,-i --format="%i|%a|%j|%u|%t|%q|%M|%D|%b|%R")
            current_now=$(date +%s)
            declare -A NEW_DB
            output_buffer=""
            while IFS='|' read -r id acct name user st qos raw_time nodes gres reason; do
                [[ -z "$id" ]] && continue
                fixed_start=0
                if [[ "$st" == "PD" ]]; then
                    if [[ -n "${JOB_DB[$id]}" && "${JOB_DB[$id]}" != "0" ]]; then fixed_start=${JOB_DB[$id]}
                    else fixed_start=$current_now; fi
                elif [[ "$st" == "R" ]]; then
                    if [[ "$raw_time" != "N/A" ]]; then
                        seconds_used=$(parse_slurm_to_seconds "$raw_time")
                        fixed_start=$((current_now - seconds_used))
                    else fixed_start=$current_now; fi
                fi
                NEW_DB[$id]=$fixed_start
                output_buffer+="${id}|${acct}|${name}|${user}|${st}|${fixed_start}|${qos}|${nodes}|${gres}|${reason}"$'\n'
            done <<< "$raw_data"
            > "$STATE_DB"
            for jid in "${!NEW_DB[@]}"; do echo "$jid ${NEW_DB[$jid]}" >> "$STATE_DB"; done
            echo -n "$output_buffer" > "${RENDER_FILE}.tmp"
            mv "${RENDER_FILE}.tmp" "$RENDER_FILE"
            # --- Node GPU Usage ---
            _nraw=$(squeue --noheader -t R --format="%u|%b|%N|%M|%i" 2>/dev/null)
            declare -A _NU _NL
            while IFS='|' read -r _u _b _n _m _id; do
                [[ -z "$_u" ]] && continue
                _g=$(echo "$_b" | grep -oP 'gpu[^:]*:\K[0-9]+')
                [[ -z "$_g" ]] && continue
                while read -r _h; do
                    [[ -z "$_h" ]] && continue
                    # edm is a burner account to hold GPUs; count its cards as free
                    [[ "$_u" != "edm" ]] && _NU[$_h]=$(( ${_NU[$_h]:-0} + _g ))
                    : ${_NU[$_h]:=0}
                    _NL[$_h]+="${_g} ${_id} ${_m} ${_u},"
                done < <(scontrol show hostnames "$_n" 2>/dev/null)
            done <<< "$_nraw"
            _nbuf=""
            for _h in $(printf '%s\n' "${!_NU[@]}" | sort); do
                _f=$(( 8 - ${_NU[$_h]} )); [[ $_f -lt 0 ]] && _f=0
                # Sort jobs by weight (gpu_count * runtime_seconds) descending
                _sorted=""
                IFS=',' read -ra _entries <<< "${_NL[$_h]}"
                _wlist=""
                for _e in "${_entries[@]}"; do
                    [[ -z "${_e// }" ]] && continue
                    read -r _eg _eid _et _eu <<< "$_e"
                    _es=$(parse_slurm_to_seconds "$_et" 2>/dev/null || echo 0)
                    _ew=$(( _eg * _es ))
                    _wlist+="${_ew} ${_e}"$'\n'
                done
                _sorted=$(echo -n "$_wlist" | sort -t' ' -k1 -rn | sed 's/^[^ ]* //')
                _slist=""
                while IFS= read -r _sl; do
                    [[ -z "$_sl" ]] && continue
                    _slist+="${_sl},"
                done <<< "$_sorted"
                _nbuf+="${_h}|${_NU[$_h]}|${_f}|${_slist}"$'\n'
            done
            echo -n "$_nbuf" > "${NODE_FILE}.tmp"
            mv "${NODE_FILE}.tmp" "$NODE_FILE"
        ) 200>"$LOCK_FILE"
        sleep 2
    done
}
backend_loop &
BACKEND_PID=$!
# ==============================================================================
# 3. GPU PROBE LOGIC
# ==============================================================================
check_gpu() {
    local c=$1; local f=$2
    # Submit with exact name for filtering. Short time limit.
    cd "$HOME" 2>/dev/null
    local j=$(sbatch --parsable --job-name=monitor_probe --account=rl --partition=compute --qos=high \
        --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:$c --mem=100G --time=00:02:00 \
        --output=/dev/null --error=/dev/null --wrap="sleep 5" 2>/dev/null)
    
    if [[ -z "$j" ]]; then echo "0" > "$f"; return; fi
    
    local start=$(date +%s)
    
    while true; do
        # STRICT 10s Timeout check
        local now=$(date +%s)
        if (( now - start > 10 )); then 
            scancel --quiet "$j"
            echo "0" > "$f"; return
        fi
        
        local s=$(squeue -j "$j" --noheader --format="%t")
        if [[ "$s" == "R" ]]; then 
            echo "1" > "$f"; scancel --quiet "$j"; return
        elif [[ -z "$s" ]]; then
             # Job gone. Assume 0.
             echo "0" > "$f"; return
        fi
        sleep 1
    done
}
run_scan() {
    echo "Querying..." > "$GPU_FILE"
    
    # 1. Parallel Check (1 and 4)
    rm -f "$HOME/.gpu1" "$HOME/.gpu4"
    check_gpu 1 "$HOME/.gpu1" & 
    check_gpu 4 "$HOME/.gpu4" & 
    wait
    
    local r1=$(cat "$HOME/.gpu1" 2>/dev/null || echo 0)
    local r4=$(cat "$HOME/.gpu4" 2>/dev/null || echo 0)
    
    # 2. Sequential Check
    if [[ "$r4" == "1" ]]; then
        check_gpu 8 "$HOME/.gpu8"
        if [[ "$(cat "$HOME/.gpu8")" == "1" ]]; then echo "8 (Max)" > "$GPU_FILE"
        else echo "4" > "$GPU_FILE"; fi
    elif [[ "$r1" == "1" ]]; then
        check_gpu 2 "$HOME/.gpu2"
        if [[ "$(cat "$HOME/.gpu2")" == "1" ]]; then echo "2" > "$GPU_FILE"
        else echo "1" > "$GPU_FILE"; fi
    else 
        echo "0" > "$GPU_FILE"
    fi
    rm -f "$HOME"/.gpu*
}
# ==============================================================================
# 4. FRONTEND RENDERER
# ==============================================================================
last_job_hash=""
format_sec() {
    local s=$1; if (( s < 0 )); then s=0; fi
    local d=$((s/86400)); local h=$(((s%86400)/3600)); local m=$(((s%3600)/60)); local sec=$((s%60))
    if (( d > 0 )); then printf "%d-%02d:%02d:%02d" $d $h $m $sec
    else printf "%02d:%02d:%02d" $h $m $sec; fi
}
_ctr() { local t="$1" w=$2 l=${#1}; local lp=$(( (w-l)/2 )); (( lp<0 )) && lp=0; local rp=$((w-l-lp)); (( rp<0 )) && rp=0; printf "%${lp}s%s%${rp}s" "" "$t" ""; }
while true; do
    read -t "$REFRESH_RATE" -n 1 -s key
    if [[ "$key" == "q" ]]; then cleanup; fi
    if [[ "$key" == "f" ]]; then
        if ! squeue --me --noheader --name=monitor_probe | grep -q .; then run_scan & fi
    fi
    if [[ "$key" == "u" ]]; then
        echo "Querying..." > "$SHARED_FILE"
        (_r=$(du -sh "/shared_work/$(whoami)/" 2>/dev/null | awk '{print $1}'); echo "${_r:-N/A}" > "${SHARED_FILE}.tmp"; mv "${SHARED_FILE}.tmp" "$SHARED_FILE") &
    fi
    if [[ "$key" == "i" ]]; then
        echo "Querying..." > "$HOME_FILE"
        (_r=$(du -sh "$HOME/" 2>/dev/null | awk '{print $1}'); echo "${_r:-N/A}" > "${HOME_FILE}.tmp"; mv "${HOME_FILE}.tmp" "$HOME_FILE") &
    fi
    now=$(date +%s)
    if [[ -f "$RENDER_FILE" ]]; then raw_content=$(cat "$RENDER_FILE"); else raw_content=""; fi
    if [[ -f "$GPU_FILE" ]]; then gpu_status=$(cat "$GPU_FILE"); else gpu_status="-"; fi
    if [[ -f "$SHARED_FILE" ]]; then shared_status=$(cat "$SHARED_FILE"); else shared_status="-"; fi
    if [[ -f "$HOME_FILE" ]]; then home_status=$(cat "$HOME_FILE"); else home_status="-"; fi
    # -- Invalidation: reset when real (non-probe) job list changes --
    cur_job_hash=$(echo "$raw_content" | grep -v "monitor_probe" | awk -F'|' '{print $1 $5}' | md5sum)
    if [[ -n "$last_job_hash" && "$cur_job_hash" != "$last_job_hash" ]]; then
        if [[ "$gpu_status" != "Querying..." ]]; then
            echo "-" > "$GPU_FILE"; gpu_status="-"
        fi
        if [[ "$shared_status" != "Querying..." ]]; then
            echo "-" > "$SHARED_FILE"; shared_status="-"
        fi
        if [[ "$home_status" != "Querying..." ]]; then
            echo "-" > "$HOME_FILE"; home_status="-"
        fi
    fi
    last_job_hash="$cur_job_hash"
    # -- CONSTRUCT BUFFER --
    screen_buf=""
    width=$(tput cols)
    bar=$(printf '%*s' "$width" '' | tr ' ' '-')
    # 1. Header
    screen_buf+="${Bold}SLURM MONITOR${Reset} | ${C}$(hostname)${Reset} | $(date '+%Y-%m-%d %H:%M:%S')${ClearLine}\n"
    screen_buf+="${Dim}${bar}${Reset}${ClearLine}\n"
    # 2. Pre-process Lists
    my_tasks_str=""
    cluster_queue_str=""
    my_count=0
    fmt_my="%-12s %-10s %-20s %-10s %-4s %10s %-5s %-6s %-5s %s"
    fmt_cl="%-12s %-10s %-20s %-10s %-4s %-8s %10s %-5s %-6s %s"
    
    IFS=$'\n' read -d '' -r -a lines <<< "$raw_content"

    # -- Pre-pass: for PD jobs with same user+card (excluding mine), only track smallest job ID --
    _me=$(whoami)
    declare -A _pd_rep
    declare -A _pd_skip
    for line in "${lines[@]}"; do
        [[ -z "$line" ]] && continue
        IFS='|' read -r _pid _pa _pn _pu _ps _pt _pq _pno _pg _pr <<< "$line"
        [[ "$_ps" != "PD" ]] && continue
        [[ "$_pu" == "$_me" ]] && continue
        _pc=$(echo "$_pg" | grep -oP 'gpu[^:]*:\K[0-9]+'); : ${_pc:="-"}
        _pkey="${_pu}|${_pc}"
        if [[ -n "${_pd_rep[$_pkey]}" ]]; then
            _prev_id=${_pd_rep[$_pkey]}
            _pid_num=${_pid%%_*}; _prev_num=${_prev_id%%_*}
            if (( _pid_num < _prev_num )); then
                _pd_skip[$_prev_id]=1; _pd_rep[$_pkey]=$_pid
            else
                _pd_skip[$_pid]=1
            fi
        else
            _pd_rep[$_pkey]=$_pid
        fi
    done

    # -- Pre-pass: lookup ReqNodeList for my jobs via scontrol --
    declare -A _my_reqn
    for line in "${lines[@]}"; do
        [[ -z "$line" ]] && continue
        IFS='|' read -r _rid _ _ _ru _ _ _ _ _ _ <<< "$line"
        [[ "$_ru" != "$_me" ]] && continue
        _rnl=$(scontrol show job "$_rid" 2>/dev/null | grep -oP 'ReqNodeList=\K\S+')
        if [[ -z "$_rnl" || "$_rnl" == "(null)" ]]; then
            _my_reqn[$_rid]="-"
        else
            _nd=$(echo "$_rnl" | grep -oP '[0-9]+$')
            _my_reqn[$_rid]="${_nd: -2}"
        fi
    done

    for line in "${lines[@]}"; do
        [[ -z "$line" ]] && continue
        IFS='|' read -r id acct name user st start_ts qos nodes gres reason <<< "$line"
        
        # Filter out pending jobs from user edm
        [[ "$user" == "edm" && "$st" == "PD" ]] && continue

        diff=$((now - start_ts))
        time_disp=$(format_sec $diff)
        color=$Dim
        if [[ "$st" == "R" ]]; then
            if [[ "$user" == "$(whoami)" ]]; then color=$G; else color=$W; fi
        elif [[ "$st" == "PD" ]]; then
            if [[ "$user" == "$(whoami)" ]]; then color=$Y; else color=$Dim; fi
            # Show pending time for all, except: ReqNodeNotAvail/Dependency reasons, or duplicate user+card (not smallest ID)
            if [[ "$reason" == *"ReqNodeNotAvail"* ]] || [[ "$reason" == *"Dependency"* ]] || [[ -n "${_pd_skip[$id]}" ]]; then
                time_disp="--:--"
            fi
        elif [[ "$st" == "CG" ]]; then color=$R; fi
        
        # Extract GPU count from gres (e.g. "gpu:8" or "gpu:a100:4")
        _card=$(echo "$gres" | grep -oP 'gpu[^:]*:\K[0-9]+')
        : ${_card:="-"}
        # Truncate reason if >25 chars, preserve closing )
        if (( ${#reason} > 25 )); then
            if [[ "$reason" == *")" ]]; then reason="${reason:0:21}...)"
            else reason="${reason:0:22}..."; fi
        fi
        cl_row_str=$(printf "$fmt_cl" "${id:0:12}" "${acct:0:10}" "${name:0:20}" "${user:0:10}" "$st" "${qos:0:8}" "$time_disp" "$nodes" "$_card" "${reason}")
        cluster_queue_str+="${color}${cl_row_str}${Reset}${ClearLine}\n"
        if [[ "$user" == "$_me" ]]; then
            _rn="${_my_reqn[$id]:-"-"}"
            my_row_str=$(printf "$fmt_my" "${id:0:12}" "${acct:0:10}" "${name:0:20}" "${user:0:10}" "$st" "$time_disp" "$nodes" "$_card" "$_rn" "${reason}")
            my_tasks_str+="${color}${my_row_str}${Reset}${ClearLine}\n"
            ((my_count++))
        fi
    done
    unset _pd_rep _pd_skip _my_reqn
    # 3. My Tasks
    screen_buf+="${Bold}${C}MY TASKS ($(whoami))${Reset}${ClearLine}\n"
    header_my=$(printf "$fmt_my" "JOBID" "ACCOUNT" "NAME" "USER" "ST" "TIME" "NODES" "CARD" "REQN" "REASON")
    header_cl=$(printf "$fmt_cl" "JOBID" "ACCOUNT" "NAME" "USER" "ST" "QOS" "TIME" "NODES" "CARD" "REASON")
    screen_buf+="${Bold}$header_my${Reset}${ClearLine}\n"
    if [[ -z "$my_tasks_str" ]]; then screen_buf+="${Dim}No active tasks.${Reset}${ClearLine}\n";
    else screen_buf+="$my_tasks_str"; fi
    screen_buf+="${ClearLine}\n"
    
    # 4. Status Table (GPU / Shared / Home)
    _sw=18
    _sbar=$(printf '%.0s─' $(seq 1 $_sw))
    screen_buf+="┌${_sbar}┬${_sbar}┬${_sbar}┐${ClearLine}\n"
    screen_buf+="│${Bold}$(_ctr "Available GPUs" $_sw)${Reset}│${Bold}$(_ctr "Shared Work" $_sw)${Reset}│${Bold}$(_ctr "Home Dir" $_sw)${Reset}│${ClearLine}\n"
    screen_buf+="├${_sbar}┼${_sbar}┼${_sbar}┤${ClearLine}\n"
    if [[ "$gpu_status" == "Querying..." ]]; then _gc=$Y; elif [[ "$gpu_status" == "-" ]]; then _gc=$Dim; else _gc=$G; fi
    screen_buf+="│${_gc}${Bold}$(_ctr "$gpu_status" $_sw)${Reset}│"
    if [[ "$shared_status" == "Querying..." ]]; then _sc=$Y; elif [[ "$shared_status" == "-" ]]; then _sc=$Dim; else _sc=$C; fi
    screen_buf+="${_sc}${Bold}$(_ctr "$shared_status" $_sw)${Reset}│"
    if [[ "$home_status" == "Querying..." ]]; then _hc=$Y; elif [[ "$home_status" == "-" ]]; then _hc=$Dim; else _hc=$C; fi
    screen_buf+="${_hc}${Bold}$(_ctr "$home_status" $_sw)${Reset}│${ClearLine}\n"
    screen_buf+="└${_sbar}┴${_sbar}┴${_sbar}┘${ClearLine}\n"
    # 4b. Node GPU Table (framed, nvitop-style)
    node_table_lines=0
    if [[ -f "$NODE_FILE" ]] && [[ -s "$NODE_FILE" ]]; then
        _nnames=(); _nused=(); _nfree=(); _nusers=()
        while IFS='|' read -r _nn _nu _nf _nul; do
            [[ -z "$_nn" ]] && continue
            _short=$(echo "$_nn" | grep -oP '[0-9]+$')
            _nnames+=("${_short:-$_nn}"); _nused+=("$_nu"); _nfree+=("$_nf"); _nusers+=("$_nul")
        done < "$NODE_FILE"
        _nc=${#_nnames[@]}; _lw=8
        # Dynamic width: full _cw=22 (1+2+1+7+1+10), compact _cw=14 (1+2+1+10) no jobid
        _cw_full=22; _cw_compact=14
        _tbl_full=$(( _lw + 2 + _nc * (_cw_full + 1) ))
        _compact=0; _cw=$_cw_full
        if (( _tbl_full > width )); then _compact=1; _cw=$_cw_compact; fi
        # Build heavy borders
        _hbar_l=$(printf '%.0s━' $(seq 1 $_lw)); _hbar_c=$(printf '%.0s━' $(seq 1 $_cw))
        _bdr_top="┏${_hbar_l}"; _bdr_mid="┣${_hbar_l}"; _bdr_bot="┗${_hbar_l}"
        for (( _i=0; _i<_nc; _i++ )); do
            _bdr_top+="┳${_hbar_c}"; _bdr_mid+="╋${_hbar_c}"; _bdr_bot+="┻${_hbar_c}"
        done
        _bdr_top+="┓"; _bdr_mid+="┫"; _bdr_bot+="┛"
        # ── Top border ──
        screen_buf+="${Bold}${_bdr_top}${Reset}${ClearLine}\n"; ((node_table_lines++))
        # ── Node row (background fill by free GPUs) ──
        _row="┃${Bold}$(_ctr "Node" $_lw)${Reset}"
        for _i in "${!_nnames[@]}"; do
            _f=${_nfree[$_i]}
            if (( _f >= 6 )); then _bg=$BgG; elif (( _f >= 3 )); then _bg=""; elif (( _f >= 1 )); then _bg=$BgY; else _bg=$BgR; fi
            _row+="┃${_bg}${W}${Bold}$(_ctr "${_nnames[$_i]}" $_cw)${Reset}"
        done
        _row+="┃"; screen_buf+="${_row}${ClearLine}\n"; ((node_table_lines++))
        # ── Separator ──
        screen_buf+="${Bold}${_bdr_mid}${Reset}${ClearLine}\n"; ((node_table_lines++))
        # ── Free row ──
        _row="┃${Bold}$(_ctr "Free" $_lw)${Reset}"
        for _i in "${!_nfree[@]}"; do
            _f=${_nfree[$_i]}
            if (( _f > 0 )); then _fc=$G; else _fc=$Dim; fi
            _row+="┃${_fc}${Bold}$(_ctr "${_nfree[$_i]}" $_cw)${Reset}"
        done
        _row+="┃"; screen_buf+="${_row}${ClearLine}\n"; ((node_table_lines++))
        # ── Separator ──
        screen_buf+="${Bold}${_bdr_mid}${Reset}${ClearLine}\n"; ((node_table_lines++))
        # ── Jobs rows (left-aligned sub-table) ──
        declare -a _urows; _max_ur=0
        for _i in "${!_nusers[@]}"; do
            IFS=',' read -ra _ues <<< "${_nusers[$_i]}"; _cnt=0
            for _ue in "${_ues[@]}"; do
                [[ -z "${_ue// }" ]] && continue
                _urows[$((_i * 100 + _cnt))]="$_ue"; ((_cnt++))
            done
            (( _cnt > _max_ur )) && _max_ur=$_cnt
        done
        for (( _r=0; _r<_max_ur; _r++ )); do
            if [[ $_r -eq 0 ]]; then _row="┃${Bold}$(_ctr "Jobs" $_lw)${Reset}"
            else _row="┃$(_ctr "" $_lw)"; fi
            for _i in "${!_nnames[@]}"; do
                _ue="${_urows[$((_i * 100 + _r))]}"
                if [[ -n "$_ue" ]]; then
                    read -r _jg _jid _jt _ju <<< "$_ue"
                    _gp=$(printf "%-2s" "$_jg")
                    _mine=0; [[ "$_ju" == "$(whoami)" ]] && _mine=1
                    if [[ $_compact -eq 0 ]]; then
                        _ip=$(printf "%-7s" "$_jid"); _tp=$(printf "%10s" "$_jt")
                        if [[ $_mine -eq 1 ]]; then
                            _cell="${BgW}${Blk} ${Bold}${_gp}${Reset}${BgW}${Blk} ${_ip} ${_tp}"
                        else
                            _cell=" ${Bold}${_gp}${Reset} ${_ip} ${_tp}"
                        fi
                    else
                        _tp=$(printf "%10s" "$_jt")
                        if [[ $_mine -eq 1 ]]; then
                            _cell="${BgW}${Blk} ${Bold}${_gp}${Reset}${BgW}${Blk} ${_tp}"
                        else
                            _cell=" ${Bold}${_gp}${Reset} ${_tp}"
                        fi
                    fi
                else
                    _cell=$(printf "%-${_cw}s" "")
                fi
                # Strip ANSI for length, then pad/truncate
                _plain=$(echo -ne "$_cell" | sed 's/\x1b\[[0-9;]*m//g')
                _plen=${#_plain}
                if (( _plen < _cw )); then
                    _pad_str=$(printf "%$((_cw - _plen))s" "")
                    if [[ -n "$_ue" && $_mine -eq 1 ]]; then
                        _cell+="${BgW}${_pad_str}"
                    else
                        _cell+="$_pad_str"
                    fi
                fi
                if [[ -n "$_ue" && $_mine -eq 1 ]]; then _cell+="${Reset}"; fi
                _row+="┃${_cell}"
            done
            _row+="┃"; screen_buf+="${_row}${ClearLine}\n"; ((node_table_lines++))
        done
        unset _urows
        # ── Bottom border ──
        screen_buf+="${Bold}${_bdr_bot}${Reset}${ClearLine}\n"; ((node_table_lines++))
        screen_buf+="${ClearLine}\n"; ((node_table_lines++))
    fi
    # 5. Cluster Queue
    # Calculate exact vertical space available
    # Used lines so far: Header(2) + MyHead(2) + MyCount + Spacer(1) + StatusTable(5) + NodeTable + Spacer(1) + ClusHead(2)
    [[ $my_count -eq 0 ]] && my_count=1
    used_vertical=$(( 2 + 2 + my_count + 1 + 5 + node_table_lines + 1 + 2 ))
    total_screen_lines=$(tput lines)
    
    # We enforce leaving 1 line at the absolute bottom for the footer
    lines_for_queue=$(( total_screen_lines - used_vertical - 1 ))
    [[ $lines_for_queue -lt 1 ]] && lines_for_queue=1
    screen_buf+="${Bold}${B}CLUSTER QUEUE${Reset}${ClearLine}\n"
    screen_buf+="${Bold}$header_cl${Reset}${ClearLine}\n"
    if [[ -z "$cluster_queue_str" ]]; then
        screen_buf+="${Dim}Queue empty.${Reset}${ClearLine}\n"
    else
        # Truncate string to exact line count to prevent pushing footer off
        subset=$(echo -ne "$cluster_queue_str" | head -n "$lines_for_queue")
        screen_buf+="$subset"
    fi
    # 6. Footer: always on the absolute last line
    footer="${Bold}${G}q${Reset}Quit ${Bold}${G}f${Reset}Check GPUs ${Bold}${G}u${Reset}Shared Work Usage ${Bold}${G}i${Reset}Home Usage${ClearLine}"
    # Atomic Print: pad blank lines to footer, no tput ed needed
    tput cup 0 0
    # Count lines in screen_buf
    _buf_lines=$(echo -ne "$screen_buf" | wc -l)
    _pad=$(( total_screen_lines - 1 - _buf_lines ))
    for (( _p=0; _p<_pad; _p++ )); do screen_buf+="${ClearLine}\n"; done
    screen_buf+="${footer}"
    echo -ne "$screen_buf"
done