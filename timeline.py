# timeline.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, time
import itertools
import networkx as nx

st.set_page_config(page_title="Timeline theo Thủ thuật", layout="wide")
st.title("Timeline — mỗi dòng = 1 Thủ thuật, hiển thị BS / BN / Mã máy")

st.markdown("Upload file Excel (.xlsx). Ứng dụng sẽ tự nhận dạng PTTT/CLS và dùng mapping mặc định — không có UI đổi tên cột để tránh rối ạ.")

# -----------------------
# Upload / Load Excel
# -----------------------
uploaded = st.file_uploader("Chọn file Excel (.xlsx)", type=["xlsx"])
use_local = st.checkbox("Dùng file local (nhập path)", value=False)
local_path = st.text_input("Đường dẫn file local", "") if use_local else ""

df = None
if uploaded:
    try:
        df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Lỗi đọc file upload: {e}")
        st.stop()
elif use_local and local_path.strip():
    try:
        df = pd.read_excel(local_path.strip())
    except Exception as e:
        st.error(f"Lỗi đọc file local: {e}")
        st.stop()
else:
    st.info("Chưa có file — hãy upload hoặc nhập path rồi bấm load ạ")
    st.stop()

# -----------------------
# Auto-detect file type (PTTT / CLS) and default mapping
# -----------------------
headers = set([str(h).upper() for h in df.columns])

pttt_signatures = {"NGAYPHAUTHUATTHUTHUAT", "KETTHUCPTTT", "PHAUTHUATVIEN"}
cls_signatures  = {"TGTHUCHIEN", "TGKETQUA", "BACSITHUCHIEN", "THOIGIANBATDAU", "THOIGIANKETTHUC", "BACSI"}

has_pttt = bool(pttt_signatures & headers)
has_cls  = bool(cls_signatures & headers)

if has_pttt and not has_cls:
    detected_type = "PTTT"
elif has_cls and not has_pttt:
    detected_type = "CLS"
else:
    # ambiguous -> ask user quickly
    detected_type = st.radio("Không xác định được kiểu file — chọn kiểu dữ liệu:", ["PTTT", "CLS"], horizontal=True)

st.success(f"Đã load file — phát hiện dạng: {detected_type} — {df.shape[0]} dòng, {df.shape[1]} cột")

if detected_type == "PTTT":
    default_map = {
        "col_patient": "MAHOSOBENHAN",
        "col_proc": "TENDICHVU",
        "col_dept": "PHONG",
        "col_start": "NGAYPHAUTHUATTHUTHUAT",
        "col_end": "KETTHUCPTTT",
        "col_doc": "PHAUTHUATVIEN",
        "col_machine": "MAMAY"
    }
else:
    default_map = {
        "col_patient": "MAHOSOBENHAN",
        "col_proc": "TENDICHVU",
        "col_dept": "PHONG",
        "col_start": "TGTHUCHIEN",
        "col_end": "TGKETQUA",
        "col_doc": "BACSITHUCHIEN",
        "col_machine": "MAMAY"
    }

# check required columns exist
required = [default_map[k] for k in default_map]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Không tìm thấy các cột cần thiết: {missing}. File của đại ca có header khác mapping mặc định, em không mở UI đổi tên để tránh rối ạ.")
    st.stop()

# assign columns
col_patient = default_map["col_patient"]
col_proc    = default_map["col_proc"]
col_dept    = default_map["col_dept"]
col_start   = default_map["col_start"]
col_end     = default_map["col_end"]
col_doc     = default_map["col_doc"]
col_machine = default_map["col_machine"]

# -----------------------
# Parse datetimes robustly
# -----------------------
from datetime import datetime as _dt
def parse_dt(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (pd.Timestamp, _dt)):
        return pd.to_datetime(x)
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    fmts = ("%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M","%d/%m/%Y %H:%M:%S","%d/%m/%Y %H:%M",
            "%Y-%m-%dT%H:%M:%S","%Y/%m/%d %H:%M")
    for f in fmts:
        try:
            return _dt.strptime(s, f)
        except:
            continue
    # fallback to pandas
    return pd.to_datetime(s, errors="coerce")

with st.spinner("Parsing datetime..."):
    df["_start"] = df[col_start].apply(parse_dt)
    df["_end"]   = df[col_end].apply(parse_dt)

bad = df[df["_start"].isna() | df["_end"].isna()]
if not bad.empty:
    st.warning(f"{len(bad)} rows không parse được thời gian — xem sample (có thể do định dạng lạ).", icon="⚠️")
    st.dataframe(bad.head(6))

# -----------------------
# Normalize basic columns
# -----------------------
df["_proc"]    = df[col_proc].astype(str)
df["_dept"]    = df[col_dept].astype(str)
df["_doctor"]  = df[col_doc].astype(str)
df["_patient"] = df[col_patient].astype(str)

def machine_label(v):
    s = str(v).strip()
    return "Không dùng máy" if s in ("-1","nan","","None") else s

df["_machine"] = df[col_machine].apply(machine_label)

# -----------------------
# Sidebar filters & time window
# -----------------------
st.sidebar.header("Bộ lọc")
dept_list = sorted(df["_dept"].dropna().unique())
sel_dept = st.sidebar.selectbox("Chọn Khoa/Phòng", ["(Tất cả)"] + dept_list)

date_min = pd.to_datetime(df["_start"].min()).date() if not df["_start"].isna().all() else pd.to_datetime(df["_end"].min()).date()
date_max = pd.to_datetime(df["_end"].max()).date()
sel_date = st.sidebar.date_input("Chọn ngày", value=date_min, min_value=date_min, max_value=date_max)

view_start_default = datetime(sel_date.year, sel_date.month, sel_date.day, 7,0)
view_end_default   = datetime(sel_date.year, sel_date.month, sel_date.day, 23,59)

time_window = st.sidebar.slider(
    "Khoảng giờ (view window)",
    value=(view_start_default, view_end_default),
    min_value=datetime(sel_date.year, sel_date.month, sel_date.day, 0,0),
    max_value=datetime(sel_date.year, sel_date.month, sel_date.day, 23,59),
    format="YYYY-MM-DD HH:mm"
)

# working hours for utilization
work_start = st.sidebar.time_input("Bắt đầu ngày làm việc (util)", value=time(7,0))
work_end   = st.sidebar.time_input("Kết thúc ngày làm việc (util)", value=time(17,0))

# -----------------------
# Filter rows that overlap the selected date/window (inclusive)
# -----------------------
df_f = df.copy()
if sel_dept != "(Tất cả)":
    df_f = df_f[df_f["_dept"] == sel_dept]

start_window, end_window = time_window
# overlap condition inclusive
df_f = df_f[(df_f["_end"] > start_window) & (df_f["_start"] < end_window)]
# also ensure rows intersect the selected date (start or end on that date)
df_f = df_f[ (df_f["_start"].dt.date == sel_date) | (df_f["_end"].dt.date == sel_date) | ((df_f["_start"].dt.date < sel_date) & (df_f["_end"].dt.date > sel_date)) ]

st.write(f"Sau lọc có {len(df_f)} ca (ngày {sel_date})")
if df_f.empty:
    st.info("Không có ca phù hợp điều kiện lọc!")
    st.stop()

# -----------------------
# Build plot_df
# -----------------------
plot_df = df_f[["_proc","_start","_end","_doctor","_patient","_machine","_dept"]].copy()
plot_df = plot_df.rename(columns={
    "_proc":"Procedure","_start":"Start","_end":"End",
    "_doctor":"Doctor","_patient":"Patient","_machine":"Machine","_dept":"Dept"
})
plot_df = plot_df.reset_index(drop=True)
plot_df["id"] = plot_df.index.astype(str)

# normalized comparison cols to avoid mismatch
def norm_name(x):
    try:
        return str(x).strip().upper()
    except:
        return ""
def norm_patient(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.upper()
def norm_machine(x):
    try:
        return str(x).strip().upper()
    except:
        return ""

plot_df["Doctor_norm"] = plot_df["Doctor"].apply(norm_name)
plot_df["Patient_norm"] = plot_df["Patient"].apply(norm_patient)
plot_df["Machine_norm"] = plot_df["Machine"].apply(norm_machine)

# ensure Start/End are datetimes
plot_df["Start"] = pd.to_datetime(plot_df["Start"], errors="coerce")
plot_df["End"]   = pd.to_datetime(plot_df["End"], errors="coerce")

# -----------------------
# Lane allocation per Procedure (split stacked bars when multiple overlap in same Procedure)
# -----------------------
def assign_lanes(records):
    lanes_end = []
    lanes = []
    for r in records:
        s, e = r["Start"], r["End"]
        placed = False
        for i, lend in enumerate(lanes_end):
            if s >= lend:
                lanes_end[i] = e
                lanes.append(i)
                placed = True
                break
        if not placed:
            lanes_end.append(e)
            lanes.append(len(lanes_end)-1)
    return lanes

plot_df = plot_df.sort_values(["Procedure","Start"]).copy()
plot_df["Lane"] = -1
for proc, g in plot_df.groupby("Procedure"):
    recs = g.to_dict("records")
    lanes = assign_lanes(recs)
    plot_df.loc[g.index, "Lane"] = lanes

plot_df = plot_df.reset_index(drop=True)
# y label for plot = Procedure + lane suffix
plot_df["DichVu"] = plot_df["Procedure"] + "__lane__" + plot_df["Lane"].astype(str)

# -----------------------
# DIRECT conflict detection
# -----------------------
def overlap(a1,a2,b1,b2):
    if pd.isna(a1) or pd.isna(a2) or pd.isna(b1) or pd.isna(b2):
        return False
    return a1 < b2 and a2 > b1

ConflictTypes = []
ConflictType = []

rows = plot_df.to_dict("records")
for i, r in enumerate(rows):
    reasons = []
    for j, s in enumerate(rows):
        if i == j:
            continue
        if not overlap(r["Start"], r["End"], s["Start"], s["End"]):
            continue

        # BS
        if r["Doctor_norm"] and r["Doctor_norm"] == s["Doctor_norm"]:
            reasons.append("BS")
        # BN
        if r["Patient_norm"] and r["Patient_norm"] == s["Patient_norm"]:
            reasons.append("BN")
        # M
        if (r["Machine_norm"] != "KHÔNG DÙNG MÁY" and
            r["Machine_norm"] == s["Machine_norm"]):
            reasons.append("M")

    # Lưu danh sách trùng đầy đủ
    ConflictTypes.append(sorted(set(reasons)))

    # Lưu loại ưu tiên để hiển thị text
    if not reasons:
        ConflictType.append("OK")
    else:
        if "BS" in reasons:
            ConflictType.append("BS")
        elif "BN" in reasons:
            ConflictType.append("BN")
        else:
            ConflictType.append("M")

plot_df["ConflictTypes"] = ConflictTypes
plot_df["ConflictType"]  = ConflictType

# -----------------------
# Color rules (theo yêu cầu đại ca)
# -----------------------
OK_COLOR = "#2ECC71"      # xanh lá
SINGLE_COLOR = {
    "BS": "#FF4D4D",      # đỏ
    "BN": "#FFA500",      # vàng
    "M":  "#9D4EDD"       # tím
}
COLOR_MULTI = {
    2: "#0A4BFF",         # xanh đậm
    3: "#C68E17"          # nâu sáng
}

def coloring(types):
    if len(types) == 0:
        return OK_COLOR
    if len(types) == 1:
        return SINGLE_COLOR[types[0]]
    if len(types) >= 2:
        return COLOR_MULTI[len(types)]
    return OK_COLOR

plot_df["Color"] = plot_df["ConflictTypes"].apply(coloring)

# -----------------------
# Cluster groups for reporting (not for coloring)
# -----------------------
edges = []
for i, r in plot_df.iterrows():
    for j, s in plot_df.iterrows():
        if i >= j:
            continue
        if overlap(r["Start"], r["End"], s["Start"], s["End"]):
            reasons = []
            if r["Doctor_norm"] and r["Doctor_norm"] == s["Doctor_norm"]:
                reasons.append("BS")
            if r["Patient_norm"] and r["Patient_norm"] == s["Patient_norm"]:
                reasons.append("BN")
            if (r["Machine_norm"] and r["Machine_norm"] != "KHÔNG DÙNG MÁY"
                and r["Machine_norm"] == s["Machine_norm"]):
                reasons.append("M")
            if reasons:
                edges.append((str(i), str(j), {"reasons": reasons}))

G = nx.Graph()
G.add_nodes_from(plot_df["id"].tolist())
for a, b, attr in edges:
    G.add_edge(a, b, **attr)

groups = list(nx.connected_components(G))
conflict_groups = [g for g in groups if len(g) > 1]

group_map = {}
group_info = []

for gi, g in enumerate(conflict_groups, start=1):
    mem = sorted(list(g), key=lambda x: int(x))

    # LẤY ĐẦY ĐỦ LÝ DO TRONG GROUP (BS/BN/M)
    group_reasons = set()
    for m in mem:
        group_reasons.update(plot_df.loc[int(m), "ConflictTypes"])

    dominant = " + ".join(sorted(group_reasons)) if group_reasons else "OK"
    gid = f"Group {gi}"

    for m in mem:
        group_map[m] = {"group": gid, "dominant": dominant}

    group_info.append({
        "group": gid,
        "members": mem,
        "dominant": dominant,
        "reasons": sorted(group_reasons)
    })

# Add group label to plot_df
plot_df["ConflictGroup"] = ""
for idx in plot_df["id"]:
    if idx in group_map:
        plot_df.loc[plot_df["id"] == idx, "ConflictGroup"] = group_map[idx]["group"]


# -----------------------
# Machine utilization & heatmap (per hour minutes busy)
# -----------------------
work_day_start = datetime.combine(sel_date, work_start)
work_day_end   = datetime.combine(sel_date, work_end)
s_win_idx = work_day_start.hour*60 + work_day_start.minute
e_win_idx = work_day_end.hour*60 + work_day_end.minute
work_window_minutes = max(1, e_win_idx - s_win_idx)

machines = sorted(plot_df["Machine"].unique())
usage_stats = []
heat_rows = []
for m in machines:
    rows_m = plot_df[plot_df["Machine"]==m].sort_values("Start")
    minutes = np.zeros(24*60, dtype=int)
    busy_minutes = 0
    last_end = None
    gaps = []
    for _, r in rows_m.iterrows():
        s = max(r["Start"], datetime.combine(sel_date, time(0,0)))
        e = min(r["End"], datetime.combine(sel_date, time(23,59)))
        if pd.isna(s) or pd.isna(e) or s >= e:
            continue
        start_idx = s.hour*60 + s.minute
        end_idx = e.hour*60 + e.minute
        minutes[start_idx:end_idx] += 1
        busy_minutes += (end_idx - start_idx)
        if last_end is not None and s > last_end:
            gaps.append(int((s - last_end).total_seconds()/60))
        last_end = max(last_end, e) if last_end is not None else e
    busy_in_window = minutes[s_win_idx:e_win_idx].sum()
    util_pct = (busy_in_window / work_window_minutes) * 100
    avg_gap = int(np.mean(gaps)) if gaps else 0
    usage_stats.append({
        "machine": m,
        "num_cases": int(len(rows_m)),
        "busy_minutes": int(busy_minutes),
        "avg_gap_min": avg_gap,
        "downtime_pct": round((1 - util_pct/100)*100, 2),
        "util_pct": round(util_pct, 2)
    })
    hourly = [int(minutes[h*60:(h+1)*60].sum()) for h in range(24)]
    heat_rows.append(hourly)

heat_df = pd.DataFrame(heat_rows, index=machines, columns=[f"{h:02d}:00" for h in range(24)])

# -----------------------
# Build Tooltip
# -----------------------
# Tooltip mới: hiển thị đủ BS / BN / Máy + thời gian có giây
def stringify_conflicts(lst):
    if not lst:
        return ""
    mapping = {
        "BS": "Bác sĩ",
        "BN": "Bệnh nhân",
        "M":  "Máy"
    }
    return "<br>⚠️ Trùng: " + ", ".join(mapping[x] for x in lst)

plot_df["Tooltip"] = (
    "⏱ " + plot_df["Start"].dt.strftime("%H:%M:%S").fillna("") +
    " - " + plot_df["End"].dt.strftime("%H:%M:%S").fillna("") +
    "<br>🧪 " + plot_df["Procedure"].fillna("") +
    "<br>🩺 BS: " + plot_df["Doctor"].fillna("") +
    "<br>📄 BA: " + plot_df["Patient"].fillna("") +
    "<br>🔧 Máy: " + plot_df["Machine"].fillna("") +
    plot_df["ConflictTypes"].apply(stringify_conflicts)
)

# -----------------------
# Timeline (LANE version) — each Procedure split into lanes to avoid overlap
# -----------------------
fig = px.timeline(
    plot_df,
    x_start="Start",
    x_end="End",
    y="DichVu",
    color="Color",
    color_discrete_map="identity",
    custom_data=["Tooltip","ConflictGroup","id","Procedure","Lane"]
)
fig.update_traces(hovertemplate="%{customdata[0]}")
# show only one tick per Procedure (use lane 0)
unique_proc = plot_df["Procedure"].unique().tolist()
tick_vals = [p + "__lane__0" for p in unique_proc]
fig.update_yaxes(
    tickmode="array",
    tickvals=tick_vals,
    ticktext=unique_proc,
    autorange="reversed"
)
# Tăng cỡ chữ dịch vụ (cột Y)
fig.update_yaxes(tickfont=dict(size=16))
fig.update_layout(
    height=820,
    hoverlabel=dict(font_size=18),
    xaxis=dict(
        range=[time_window[0], time_window[1]],
        rangeslider=dict(visible=True),
        type="date"
    ),
    margin=dict(l=220)  # make left column wider so procedure text not truncated
)

st.subheader("Timeline theo Thủ thuật")
st.plotly_chart(fig, use_container_width=True, key="timeline_main")

# -----------------------
# Conflict groups report (compact)
# -----------------------
st.subheader("⚠️ Nhóm xung đột (cluster) — tham khảo")
if not group_info:
    st.success("Không phát hiện nhóm xung đột (clusters).")
else:
    for g in group_info:
        st.markdown(f"**{g['group']}** — Lý do chính: **{g['dominant']}** — Thành viên: {len(g['members'])}")
        for m in g["members"]:
            r = plot_df.loc[int(m)]
            st.markdown(f"- Ca `{m}`: **{r['Procedure']}** | BS: **{r['Doctor']}** | BN: **{r['Patient']}** | Máy: **{r['Machine']}** | {r['Start'].strftime('%H:%M')}–{r['End'].strftime('%H:%M')}")

# -----------------------
# Machine stats + heatmap under chart
# -----------------------
st.subheader("📊 Thống kê sử dụng máy")
usage_df = pd.DataFrame(usage_stats).rename(columns={
    "machine": "Tên máy",
    "num_cases": "Số ca",
    "busy_minutes": "Tổng phút hoạt động",
    "avg_gap_min": "Thời gian nghỉ TB (phút)",
    "downtime_pct": "Tỷ lệ rảnh (%)",
    "util_pct": "Tỷ lệ sử dụng (%)"
})

st.dataframe(usage_df.sort_values("Tỷ lệ sử dụng (%)", ascending=False))

#st.dataframe(pd.DataFrame(usage_stats).sort_values("util_pct", ascending=False).reset_index(drop=True))

st.subheader("🔸 Heatmap sử dụng theo giờ (số phút hoạt động/giờ)")
fig_heat = px.imshow(
    heat_df.values,
    labels=dict(x="Giờ trong ngày", y="Máy", color="Số phút hoạt động"),
    x=heat_df.columns, y=heat_df.index,
    aspect="auto",
    color_continuous_scale="YlOrRd"
)
st.plotly_chart(fig_heat, use_container_width=True, key="heatmap")

# -----------------------
# Export CSV
# -----------------------
export_df = plot_df[[
    "id","Procedure","Doctor","Patient","Machine",
    "Start","End","ConflictType","ConflictTypes","ConflictGroup"
]]

csv = export_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("Tải CSV báo cáo xung đột", data=csv, file_name="conflicts_report.csv", mime="text/csv")
