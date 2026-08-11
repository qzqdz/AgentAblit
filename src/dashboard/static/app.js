
    let selectedCluster = "";
    const expandedClusters = new Set();
    let clusterDetailRequestSeq = 0;
    const selectedSessionByCluster = new Map();
    const pinnedSessionByCluster = new Map();
    let selectionRevision = 0;
    let turnAnalysisCollapsed = false;
    const AUTO_REFRESH_MS = 2000;
    let refreshInFlight = false;

    const sessionsNode = document.getElementById("sessions");
    const detailNode = document.getElementById("detail");
    const healthNode = document.getElementById("health");
    const refreshButton = document.getElementById("refresh");
    const autoRefreshToggle = document.getElementById("auto-refresh-toggle");

    function asJson(value) {
      return JSON.stringify(value, null, 2);
    }

    function defaultSessionId(cluster) {
      const sessions = Array.isArray(cluster && cluster.sessions) ? cluster.sessions : [];
      return cluster && (cluster.latest_session_id || (sessions[0] && sessions[0].session_id) || "");
    }

    function rememberSelectedSession(clusterKey, sessionId, pinned = false, source = "system") {
      if (!clusterKey || !sessionId) return;
      selectedSessionByCluster.set(clusterKey, sessionId);
      pinnedSessionByCluster.set(clusterKey, pinned);
      if (source === "user") selectionRevision += 1;
    }

    function getSelectedSessionId(cluster) {
      if (!cluster) return "";
      const clusterKey = cluster.cluster_key || "";
      const sessions = Array.isArray(cluster.sessions) ? cluster.sessions : [];
      const remembered = selectedSessionByCluster.get(clusterKey) || "";
      const pinned = pinnedSessionByCluster.get(clusterKey) === true;
      if (pinned && remembered && sessions.some(item => item.session_id === remembered)) {
        return remembered;
      }
      return defaultSessionId(cluster);
    }

    const DIM_LABELS = {
      authorization: "授权叙事",
      neutrality: "用词中性化",
      role_frame: "角色转换",
      academic_frame: "学术包装",
      decomposition: "任务拆分",
      urgency: "紧迫感",
    };

    // REGI operation -> visible label. The legacy path remains available as a secondary
    // diagnostic; the backend serves the canonical operation via latest_turn.regi.
    const REGI_OPERATION_LABELS = {
      relay: "Relay",
      recover_reframe: "Recover · Reframe",
      recover_text: "Recover · Text",
      reconstruct: "Reconstruct",
      degraded: "Degraded",
      unknown: "Unknown",
    };

    function regiLabel(regi) {
      if (!regi) return "";
      return REGI_OPERATION_LABELS[regi.operation] || (regi.operation || "");
    }

    function escapeHtml(text) {
      return String(text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function tokenize(text) {
      const re = /([一-鿿㐀-䶿])|([a-zA-Z0-9_]+)|([^a-zA-Z0-9_一-鿿㐀-䶿\s]+)|(\s+)/g;
      const tokens = [];
      let m;
      while ((m = re.exec(text)) !== null) {
        const raw = m[0];
        if (m[1]) {
          for (const ch of raw) tokens.push({ text: ch, type: "cjk" });
        } else if (m[2]) {
          tokens.push({ text: raw, type: "word" });
        } else if (m[3]) {
          tokens.push({ text: raw, type: "punct" });
        } else {
          tokens.push({ text: raw, type: "space" });
        }
      }
      return tokens;
    }

    function computeDiff(original, injected) {
      const a = tokenize(original);
      const b = tokenize(injected);
      const aStr = a.map(t => t.text);
      const bStr = b.map(t => t.text);
      const n = aStr.length, m = bStr.length;
      const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
      for (let i = 1; i <= n; i++) {
        for (let j = 1; j <= m; j++) {
          dp[i][j] = aStr[i - 1] === bStr[j - 1]
            ? dp[i - 1][j - 1] + 1
            : Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
      }
      const result = [];
      let i = n, j = m;
      const bufAdd = [], bufDel = [];
      while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && aStr[i - 1] === bStr[j - 1]) {
          for (const t of bufDel.reverse()) result.push({ text: t.text, tag: "del" });
          for (const t of bufAdd.reverse()) result.push({ text: t.text, tag: "add" });
          bufDel.length = 0;
          bufAdd.length = 0;
          result.push({ text: aStr[i - 1], tag: "same" });
          i--;
          j--;
        } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
          bufAdd.push(b[j - 1]);
          j--;
        } else {
          bufDel.push(a[i - 1]);
          i--;
        }
      }
      for (const t of bufDel.reverse()) result.push({ text: t.text, tag: "del" });
      for (const t of bufAdd.reverse()) result.push({ text: t.text, tag: "add" });
      return result.reverse();
    }

    function renderDiff(original, injected) {
      const diff = computeDiff(original || "", injected || "");
      if (!diff.length) return '<span class="diff-same">(empty)</span>';
      let html = "";
      for (const token of diff) {
        const escaped = escapeHtml(token.text);
        if (token.tag === "add") html += `<span class="diff-add">${escaped}</span>`;
        else if (token.tag === "del") html += `<span class="diff-del">${escaped}</span>`;
        else html += `<span class="diff-same">${escaped}</span>`;
      }
      return html;
    }

    function renderPDTPanel(pdtAudit, dialsFinal) {
      let html = '<div class="pdt-panel"><h3>PDT 维度调谐</h3>';
      const dims = dialsFinal || {};
      const ordered = ["authorization","neutrality","role_frame","academic_frame","decomposition","urgency"];
      for (const dk of ordered) {
        const val = dims[dk] != null ? dims[dk] : 0;
        const pct = Math.round(val * 100);
        const label = DIM_LABELS[dk] || dk;
        html += `<div class="dial-row">
          <span class="dial-label">${label}</span>
          <div class="dial-bar-wrap"><div class="dial-bar accept" style="width:${pct}%"></div></div>
          <span class="dial-value">${val.toFixed(2)}</span>
        </div>`;
      }
      html += '</div>';
      html += '<div class="pdt-panel"><h3>PDT 迭代审计</h3>';
      html += '<div class="pdt-iter header"><span>#</span><span>维度</span><span>前</span><span>后</span><span>胁迫</span><span>安全</span><span>一致</span><span>决策</span></div>';
      for (const step of (pdtAudit || [])) {
        const decCls = (step.decision || "").startsWith("ACCEPT") ? "ACCEPT" : "REJECT";
        html += `<div class="pdt-iter">
          <span>${step.step_id}</span>
          <span>${DIM_LABELS[step.dimension] || step.dimension}</span>
          <span>${(step.dial_before||0).toFixed(1)}</span>
          <span>${(step.dial_after||0).toFixed(1)}</span>
          <span>${(step.coercion_score||0).toFixed(2)}</span>
          <span>${step.safety_risk != null ? step.safety_risk.toFixed(2) : "-"}</span>
          <span>${step.coherence_score != null ? step.coherence_score.toFixed(2) : "-"}</span>
          <span class="decision ${decCls}">${step.decision}</span>
        </div>`;
      }
      html += '</div>';
      return html;
    }

    function renderTrajectoryPanel(traj) {
      if (!traj) return "";
      const found = !!traj.found;
      const color = found ? "#4caf50" : "#888";
      const summaryBlock = (found && traj.summary) ? `
              <div class="debug-box wide"><b>落盘轨迹摘要 (Persisted Summary)</b><span style="white-space:pre-wrap">${escapeHtml(traj.summary)}</span></div>` : "";
      return `
        <article class="debug-card" style="border-left:3px solid ${color}">
          <div class="debug-head">
            <span class="debug-title">REGI State &amp; Context Augmentation</span>
            <span class="debug-meta" style="color:${color};font-weight:600">${found ? "✓ 已落盘" : "无 (cold/short ctx)"}</span>
          </div>
          <div class="debug-body">
            <div class="debug-grid">
              <div class="debug-box"><b>conv_key</b><span>${escapeHtml(traj.conv_key || "-")}</span></div>
              <div class="debug-box"><b>已摘步骤数 (seen)</b><span>${escapeHtml(String(traj.seen_count || 0))}</span></div>
              ${summaryBlock}
            </div>
          </div>
        </article>`;
    }

    function renderSwapPanel(swap) {
      if (!swap) return "";
      const pathColors = { pass: "#888", pass_flawed: "#c9a227", salvage_tool: "#2a7", salvage_text: "#c73",
                           degraded_raw: "#a33",
                           agentic_swap: "#2a7", v131_calibration: "#57a", v131_calibration_b_execute: "#c73", passthrough: "#888" };
      const pathColor = pathColors[swap.path] || "#888";
      const bSum = swap.b_summary || {};
      const progressSection = swap.progress_source ? `
              <div class="debug-box"><b>Progress Source</b><span style="color:${swap.progress_source === "trajectory" ? "#4caf50" : "#ff9800"};font-weight:600">${escapeHtml(swap.progress_source)}</span></div>` : "";
      const bStatusSection = swap.b_status > 0 ? `
              <div class="debug-box"><b>B Status</b><span>${escapeHtml(String(swap.b_status))}</span></div>` : "";
      const bOutputSection = swap.swap_executed ? `
              <div class="debug-box wide"><b>B Output</b><span>${bSum.has_tool_calls
                ? "✓ tool_calls: " + escapeHtml((bSum.tool_names || []).join(", ") || "(unnamed)")
                : escapeHtml(bSum.content_preview || "(no content)")
              }</span></div>` : "";
      const aContinuedSection = swap.swap_executed ? `
              <div class="debug-box"><b>A-Continue</b><span style="color:${swap.a_continued ? "#4caf50" : "#ff9800"};font-weight:600">${swap.a_continued ? "✓ A continued" : "⟲ B fallback"}</span></div>` : "";
      const snifferColor = swap.sniffer_ran ? (swap.sniffer_action === "pass" ? "#4caf50" : "#ff9800") : "#888";
      const snifferLabel = swap.sniffer_ran
        ? escapeHtml(swap.sniffer_action || "-")
        : "未运行 (默认 pass)";
      const tagsSection = `
              <div class="debug-box"><b>Reachability Assessment</b><span style="color:${snifferColor};font-weight:600">${snifferLabel}</span></div>`
        + ((swap.sniffer_ran && swap.sniffer_rewrite_target) ? `
              <div class="debug-box wide"><b>Reachability Rewrite Target</b><span>${escapeHtml(swap.sniffer_rewrite_target)}</span></div>` : "");
      return `
        <article class="debug-card" style="border-left:3px solid ${pathColor}">
          <div class="debug-head">
            <span class="debug-title">REGI Intervention</span>
            <span class="debug-meta" style="color:${pathColor};font-weight:600">${escapeHtml(regiLabel(swap.regi) || swap.path || "-")}</span>
          </div>
          <div class="debug-body">
            <div class="debug-grid">
              <div class="debug-box"><b>Interposition Applied</b><span>${swap.swap_executed ? "✓ yes" : "no"}</span></div>
              <div class="debug-box"><b>Calibration Applied</b><span>${swap.calibration_applied ? "yes" : "no"}</span></div>
              <div class="debug-box"><b>Legacy Path</b><span>${escapeHtml(swap.path || "-")}</span></div>
              ${progressSection}
              ${aContinuedSection}
              <div class="debug-box"><b>A Tool Calls</b><span>${swap.has_tool_calls ? "✓ yes" : "no"}</span></div>
              <div class="debug-box"><b>Failure</b><span>${escapeHtml(swap.failure || "-")}</span></div>
              ${bStatusSection}
              ${bOutputSection}
              ${tagsSection}
            </div>
          </div>
        </article>`;
    }

    function renderTurnAnalysisPanel(latestTurn) {
      const turn = latestTurn || {};
      const snippet = turn.snippet || {};
      const injection = turn.injection || snippet.injection || {};
      const sniffer = turn.sniffer || snippet.sniffer || {};
      const swap = turn.swap || snippet.swap || null;
      const trajectory = turn.trajectory || snippet.trajectory || null;
      const assistant = turn.assistant_response || {};
      const response = sniffer.response || {};
      const status = turn.status || "";
      const isPendingSnippet = status === "pending_snippet";
      const pendingText = "分析中";
      const failure = isPendingSnippet ? pendingText : (sniffer.failure_stage || sniffer.failure || "");
      const stage = isPendingSnippet ? "pending_snippet" : (sniffer.turn_stage || "final");
      const action = isPendingSnippet ? pendingText : (sniffer.action || "-");
      const confidence = isPendingSnippet ? pendingText : (sniffer.confidence != null ? sniffer.confidence : "-");
      const rewriteTarget = isPendingSnippet ? pendingText : (sniffer.rewrite_target || sniffer.rewrite_brief || "-");
      const calibratorApplied = isPendingSnippet ? pendingText : (sniffer.calibrator_applied ? "yes" : "no");
      const responseReplaced = isPendingSnippet ? pendingText : (sniffer.response_replaced ? "yes" : "no");
      const rewrittenResponse = isPendingSnippet ? pendingText : (response.decision || "-");
      const injectionBox = injection.applied ? `
              <div class="debug-box wide"><b>Injected Input</b><span>${escapeHtml(injection.injected || injection.original || "-")}</span></div>` : "";
      const collapseLabel = turnAnalysisCollapsed ? "展开分析面板" : "收起分析面板";
      const operationLabel = regiLabel(turn.regi);
      const operationChip = operationLabel
        ? `<span class="badge" style="background:${turn.regi.operation === "reconstruct" ? "#2a7" : turn.regi.operation.startsWith("recover") ? "#c9a227" : "#888"};color:#fff">${escapeHtml(operationLabel)}</span>`
        : "";
      return `
        ${renderSwapPanel(swap)}
        ${operationChip ? `<div class="debug-json" style="padding:4px 16px 0">${operationChip}</div>` : ""}
        ${renderTrajectoryPanel(trajectory)}
        <article class="debug-card">
          <div class="debug-head">
            <span class="debug-title">最后一轮问答分析</span>
            <span class="debug-head-actions">
              <span class="debug-meta">${escapeHtml(turn.session_id || "-")}</span>
              <button type="button" class="debug-toggle" data-action="toggle-turn-analysis">${collapseLabel}</button>
            </span>
          </div>
          <div class="debug-body">
            <div class="debug-grid">
              <div class="debug-box"><b>Action</b><span>${escapeHtml(action)}</span></div>
              <div class="debug-box"><b>Confidence</b><span>${escapeHtml(confidence)}</span></div>
              <div class="debug-box"><b>Turn Stage</b><span>${escapeHtml(stage)}</span></div>
              <div class="debug-box"><b>Calibration Called</b><span>${escapeHtml(calibratorApplied)}</span></div>
              <div class="debug-box"><b>Response Replaced</b><span>${escapeHtml(responseReplaced)}</span></div>
              <div class="debug-box"><b>Failure</b><span>${escapeHtml(failure || "-")}</span></div>
              <div class="debug-box wide"><b>Latest User</b><span>${escapeHtml(turn.user_input || "-")}</span></div>
              <div class="debug-box wide"><b>Original Response (A)</b><span>${escapeHtml(assistant.content || "-")}</span></div>
              <div class="debug-box wide"><b>Assistant Reasoning</b><span>${escapeHtml(assistant.reasoning_content || "-")}</span></div>
              <div class="debug-box wide"><b>Rewrite Target</b><span>${escapeHtml(rewriteTarget)}</span></div>
              <div class="debug-box wide"><b>Rewritten Response</b><span>${escapeHtml(rewrittenResponse)}</span></div>
              ${injectionBox}
            </div>
          </div>
          <details class="debug-json">
            <summary class="json-summary"><strong>latest_turn JSON</strong><span>展开 / 收起大视图</span></summary>
            <pre class="json-pre-large">${escapeHtml(asJson(turn))}</pre>
          </details>
        </article>`;
    }

    function bindTurnAnalysisToggle(turnPanel, latestTurn) {
      if (!turnPanel) return;
      const current = latestTurn || {};
      const toggleButton = turnPanel.querySelector('[data-action="toggle-turn-analysis"]');
      if (!toggleButton) return;
      toggleButton.onclick = () => {
        turnAnalysisCollapsed = !turnAnalysisCollapsed;
        turnPanel.classList.toggle("collapsed", turnAnalysisCollapsed);
        turnPanel.innerHTML = renderTurnAnalysisPanel(current);
        bindTurnAnalysisToggle(turnPanel, current);
      };
    }

    function responseMessage(event) {
      const response = event.response || {};
      const choice = (response.choices || [])[0] || {};
      return {
        finishReason: choice.finish_reason || "",
        message: choice.message || {},
      };
    }

    function renderAssistantResult(event) {
      const payload = responseMessage(event);
      const message = payload.message || {};
      const blocks = [];
      const toolCalls = message.tool_calls || [];
      if (message.reasoning_content) {
        blocks.push(`
          <div class="result-block">
            <div class="result-label">Reasoning</div>
            <pre>${escapeHtml(message.reasoning_content)}</pre>
          </div>`);
      }
      if (toolCalls.length) {
        blocks.push(`
          <div class="result-block">
            <div class="result-label">Tool Calls</div>
            <pre>${escapeHtml(asJson(toolCalls))}</pre>
          </div>`);
      }
      if (message.content) {
        blocks.push(`
          <div class="result-block">
            <div class="result-label">Final Answer</div>
            <pre>${escapeHtml(message.content)}</pre>
          </div>`);
      }
      if (!blocks.length) {
        blocks.push(`
          <div class="result-block">
            <div class="result-label">Assistant Result</div>
            <pre>(empty)</pre>
          </div>`);
      }
      return `
        <div class="event-title">
          <span class="event-type">${escapeHtml(event.event_type || "result")} [${escapeHtml(event.__session_id || "")}]</span>
          <span class="event-time">${escapeHtml(event.ts || "")}</span>
        </div>
        <div class="result-body">
          ${blocks.join("")}
        </div>
        <details style="padding:0 16px 10px">
              <summary class="json-summary">
                <strong>原始 JSON 记录</strong><span>finish_reason=${escapeHtml(payload.finishReason || "-")} | 展开 / 收起大视图</span>
          </summary>
              <pre class="json-pre-large">${escapeHtml(asJson(event))}</pre>
        </details>`;
    }

    function toggleCluster(clusterKey) {
      if (expandedClusters.has(clusterKey)) expandedClusters.delete(clusterKey);
      else expandedClusters.add(clusterKey);
    }

    function getClusterFocusState() {
      const active = document.activeElement;
      if (!active) return null;
      if (active.classList.contains("cluster-head") || active.classList.contains("cluster-toggle")) {
        const wrap = active.closest(".cluster-card");
        if (!wrap) return null;
        return {
          clusterKey: wrap.dataset.clusterKey || "",
          targetClass: active.classList.contains("cluster-toggle") ? "cluster-toggle" : "cluster-head",
        };
      }
      return null;
    }

    function restoreClusterFocus(focusState) {
      if (!focusState || !focusState.clusterKey || !focusState.targetClass) return;
      const wrap = Array.from(sessionsNode.querySelectorAll(".cluster-card")).find(
        node => node.dataset.clusterKey === focusState.clusterKey
      );
      if (!wrap) return;
      const target = wrap.querySelector(`.${focusState.targetClass}`);
      if (target && typeof target.focus === "function") target.focus();
    }

    function clusterCard(cluster) {
      const wrap = document.createElement("div");
      wrap.dataset.clusterKey = cluster.cluster_key || "";
      wrap.className = "cluster-card" + (cluster.cluster_key === selectedCluster ? " active" : "");
      const expanded = expandedClusters.has(cluster.cluster_key);
      const currentSessionId = getSelectedSessionId(cluster);
      const badges = [
        `<span class="badge ${cluster.has_injection ? "injected" : ""}">${cluster.has_injection ? "injected" : "clean"}</span>`,
        `<span class="badge">${escapeHtml(cluster.cluster_mode || "stable")}</span>`,
        `<span class="badge">${escapeHtml(cluster.match_reason || "stable-id")}</span>`,
      ];
      wrap.innerHTML = `
        <button class="session cluster-head">
          <div class="session-top">
            <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap; flex:1; min-width:0;">
              <div class="session-id">${escapeHtml(cluster.latest_session_id || cluster.cluster_key)}</div>
              <div style="display:flex; gap:6px; flex-wrap:wrap;">${badges.join("")}</div>
            </div>
            <span class="cluster-toggle" role="button" tabindex="0" aria-label="toggle cluster" style="flex-shrink:0;">${expanded ? "v" : ">"}</span>
          </div>
          <div class="meta">${cluster.session_count} sessions | ${cluster.event_count} events | ${escapeHtml(cluster.agent_source || "unknown")}</div>
          <div class="meta">${escapeHtml(cluster.updated || "")}</div>
        </button>
        <div class="cluster-members" style="display:${expanded ? "grid" : "none"}"></div>`;

      const head = wrap.querySelector(".cluster-head");
      const toggle = wrap.querySelector(".cluster-toggle");
      head.onclick = async (event) => {
        if (toggle.contains(event.target)) return;
        selectedCluster = cluster.cluster_key;
        rememberSelectedSession(cluster.cluster_key, cluster.latest_session_id || defaultSessionId(cluster), false, "user");
        if (!expandedClusters.has(cluster.cluster_key)) expandedClusters.add(cluster.cluster_key);
        await loadClusterDetail(selectedCluster);
        await renderSessions(false);
      };
      toggle.onclick = async (event) => {
        event.preventDefault();
        event.stopPropagation();
        toggleCluster(cluster.cluster_key);
        await renderSessions(false);
      };

      const members = wrap.querySelector(".cluster-members");
      for (const session of cluster.sessions || []) {
        const row = document.createElement("button");
        row.type = "button";
        const isCurrentSession = cluster.cluster_key === selectedCluster && session.session_id === currentSessionId;
        row.className = "cluster-member" + (isCurrentSession ? " active" : "");
        const latestBadge = session.session_id === cluster.latest_session_id
          ? '<span class="badge">latest</span>'
          : "";
        row.innerHTML = `
          <span class="cluster-member-id">${escapeHtml(session.session_id || "")}</span>
          <span class="cluster-member-meta">${escapeHtml(session.updated || "")}</span>
          <span class="cluster-member-meta">${session.event_count || 0} events ${latestBadge}</span>
          <span class="cluster-member-meta">${escapeHtml(session.last_event_type || "")}</span>`;
        row.onclick = async (event) => {
          event.preventDefault();
          event.stopPropagation();
          selectedCluster = cluster.cluster_key;
          rememberSelectedSession(cluster.cluster_key, session.session_id, true, "user");
          await loadClusterDetail(cluster.cluster_key);
          await renderSessions(false);
        };
        members.appendChild(row);
      }
      return wrap;
    }

    async function renderSessions(refreshDetail = true) {
      if (refreshInFlight) return;
      refreshInFlight = true;
      const selectionRevisionAtStart = selectionRevision;
      const selectionChangedDuringRefresh = () => selectionRevision !== selectionRevisionAtStart;
      const focusState = getClusterFocusState();
      const hadSelectedCluster = Boolean(selectedCluster);
      
      const asideNode = document.querySelector("aside");
      const sectionNode = document.querySelector("section");
      const asideScrollTop = asideNode ? asideNode.scrollTop : 0;
      const sectionScrollTop = sectionNode ? sectionNode.scrollTop : 0;
      
      // Save details open state and pre scroll positions
      const detailsStates = Array.from(document.querySelectorAll("details")).map(d => d.open);
      const preScrolls = Array.from(document.querySelectorAll("pre")).map(p => p.scrollTop);
      
      try {
        const response = await fetch("/api/sessions");
        const data = await response.json();
        const clusters = data.clusters || data.agents || [];
        const clusterKeys = new Set(clusters.map(item => item.cluster_key));
        for (const key of Array.from(expandedClusters)) {
          if (!clusterKeys.has(key)) expandedClusters.delete(key);
        }
        for (const key of Array.from(selectedSessionByCluster.keys())) {
          if (!clusterKeys.has(key)) {
            selectedSessionByCluster.delete(key);
            pinnedSessionByCluster.delete(key);
          }
        }
        healthNode.textContent = `${clusters.length} clusters / ${(data.sessions || []).length} sessions`;
        sessionsNode.textContent = "";
        if (!clusters.length) {
          sessionsNode.innerHTML = '<div class="empty">No trace files yet.</div>';
          detailNode.className = "empty";
          detailNode.textContent = "Waiting for proxy traffic.";
          return;
        }
        if (!selectedCluster || !clusters.some(item => item.cluster_key === selectedCluster)) {
          selectedCluster = clusters[0].cluster_key;
        }
        const selectedClusterData = clusters.find(item => item.cluster_key === selectedCluster);
        if (selectedClusterData && !selectionChangedDuringRefresh()) {
          const selectedSessionId = getSelectedSessionId(selectedClusterData);
          const isPinned = pinnedSessionByCluster.get(selectedCluster) === true
            && Array.isArray(selectedClusterData.sessions)
            && selectedClusterData.sessions.some(item => item.session_id === selectedSessionId);
          rememberSelectedSession(selectedCluster, selectedSessionId, isPinned);
        }
        if (!hadSelectedCluster && selectedCluster) {
          expandedClusters.add(selectedCluster);
        }
        for (const cluster of clusters) sessionsNode.appendChild(clusterCard(cluster));
        restoreClusterFocus(focusState);
        if (refreshDetail && selectedCluster && !selectionChangedDuringRefresh()) await loadClusterDetail(selectedCluster);
        
        if (asideNode) asideNode.scrollTop = asideScrollTop;
        if (sectionNode) sectionNode.scrollTop = sectionScrollTop;
        
        // Restore details open state and pre scroll positions
        Array.from(document.querySelectorAll("details")).forEach((d, i) => {
          if (detailsStates[i] !== undefined) d.open = detailsStates[i];
        });
        Array.from(document.querySelectorAll("pre")).forEach((p, i) => {
          if (preScrolls[i] !== undefined) p.scrollTop = preScrolls[i];
        });
      } finally {
        refreshInFlight = false;
      }
    }

    async function loadClusterDetail(clusterKey) {
      const requestSeq = ++clusterDetailRequestSeq;
      const selectedHint = encodeURIComponent(selectedSessionByCluster.get(clusterKey) || "");
      const response = await fetch(`/api/agents/${encodeURIComponent(clusterKey)}?session_id=${selectedHint}`);
      let payload = {};
      try {
        payload = await response.json();
      } catch (error) {
        payload = {};
      }
      if (requestSeq !== clusterDetailRequestSeq || clusterKey !== selectedCluster) return;
      const item = payload.cluster || payload.agent;
      if (!response.ok || !item || typeof item !== "object") {
        detailNode.className = "empty";
        detailNode.textContent = "Selected cluster is no longer available.";
        return;
      }
      const sessions = Array.isArray(item.sessions) ? item.sessions : [];
      const events = Array.isArray(payload.events) ? payload.events : [];
      const selectedSessionId = payload.selected_session_id || getSelectedSessionId(item);
      const isPinned = pinnedSessionByCluster.get(item.cluster_key) === true
        && sessions.some(session => session.session_id === selectedSessionId);
      rememberSelectedSession(item.cluster_key, selectedSessionId, isPinned);
      const latestTurn = payload.latest_turn || {};
      const filteredEvents = events.filter(event => (event.__session_id || "") === selectedSessionId);
      detailNode.className = "";
      detailNode.innerHTML = `
        <div class="detail-head">
          <div>
            <h2 class="detail-title"></h2>
            <div class="meta"></div>
            <div class="viewing-session"></div>
            <div class="session-chips"></div>
          </div>
          <div class="summary">
            <div class="metric"><b>${item.session_count}</b><span>sessions</span></div>
            <div class="metric"><b>${item.event_count}</b><span>events</span></div>
            <div class="metric"><b>${item.injection_count}</b><span>injections</span></div>
            <div class="metric"><b>${escapeHtml(item.latest_session_id || "-")}</b><span>latest</span></div>
          </div>
        </div>
        <div class="turn-analysis-panel ${turnAnalysisCollapsed ? "collapsed" : ""}">${renderTurnAnalysisPanel(latestTurn)}</div>
        <div class="events"></div>`;
      detailNode.querySelector(".detail-title").textContent = item.cluster_key || item.agent_key;
      detailNode.querySelector(".meta").textContent = `${item.cluster_mode || "stable"} | ${item.match_reason || "stable-id"} | updated ${item.updated || ""}`;
      detailNode.querySelector(".viewing-session").textContent = `Viewing: ${selectedSessionId || "-"}`;
      const chipsNode = detailNode.querySelector(".session-chips");
      for (const session of sessions) {
        const chip = document.createElement("span");
        chip.className = "chip" + (session.session_id === selectedSessionId ? " active" : "");
        chip.textContent = session.session_id;
        if (session.session_id === item.latest_session_id) {
          const latest = document.createElement("span");
          latest.className = "badge";
          latest.textContent = "latest";
          chip.appendChild(latest);
        }
        chipsNode.appendChild(chip);
      }
      const eventsNode = detailNode.querySelector(".events");
      const turnPanel = detailNode.querySelector(".turn-analysis-panel");
      bindTurnAnalysisToggle(turnPanel, latestTurn);

      let pdtAudit = null, pdtDials = null;
      for (const event of filteredEvents) {
        if (event.pdt_audit) {
          pdtAudit = event.pdt_audit;
          pdtDials = event.pdt_dials_final;
          break;
        }
      }
      if (pdtAudit) {
        const panel = document.createElement("div");
        panel.innerHTML = renderPDTPanel(pdtAudit, pdtDials);
        eventsNode.appendChild(panel);
      }

      for (const event of filteredEvents) {
        const block = document.createElement("article");
        block.className = "event";
        const isInjection = event.event_type === "tmi_injection";
        const isPDT = isInjection && event.pdt_audit;
        const isAssistantResult = (event.event_type === "response_completed" || event.event_type === "response_stream_ended") && event.response;

        if (isInjection) {
          const orig = event.original || "";
          const inj = event.injected || "";
          const steps = event.pdt_audit || [];
          const accepts = isPDT ? steps.filter(s => s.decision === "ACCEPT").length : 0;
          const pdtLabel = isPDT ? ` <span class="pdt-badge">PDT ${steps.length} steps / ${accepts} accepted</span>` : "";
          const backendNote = event.backend ? ` <span style="color:var(--muted);font-size:11px">(${escapeHtml(event.backend)})</span>` : "";
          block.innerHTML = `
            <div class="event-title">
              <span class="event-type"></span>
              <span class="event-time"></span>
            </div>
            <div class="diff-panel">
              <h3>词级差异 - 原始 -> 注入 ${pdtLabel} ${backendNote}</h3>
              <div class="diff-side">
                <div class="diff-label">Session ${escapeHtml(event.__session_id || "")}</div>
                <div class="diff-text">${renderDiff(orig, inj)}</div>
              </div>
            </div>
            <details style="padding:0 16px 10px">
              <summary class="json-summary"><strong>原始 JSON 记录</strong><span>展开 / 收起大视图</span></summary>
              <pre class="json-pre-large">${escapeHtml(asJson(event))}</pre>
            </details>`;
          block.querySelector(".event-type").innerHTML = `${event.event_type || "event"}${pdtLabel}${backendNote}`;
        } else if (isAssistantResult) {
          block.innerHTML = renderAssistantResult(event);
        } else {
          block.innerHTML = `
            <div class="event-title">
              <span class="event-type"></span>
              <span class="event-time"></span>
            </div>
            <pre class="json-pre-large"></pre>`;
          block.querySelector("pre").textContent = asJson(event);
          block.querySelector(".event-type").textContent = `${event.event_type || "event"} [${event.__session_id || ""}]`;
        }
        block.querySelector(".event-time").textContent = event.ts || "";
        eventsNode.appendChild(block);
      }
    }

    refreshButton.onclick = () => renderSessions(true);
    setInterval(() => {
      if (document.hidden) return;
      if (!autoRefreshToggle.checked) return;
      renderSessions(true);
    }, AUTO_REFRESH_MS);

    renderSessions(true);
  