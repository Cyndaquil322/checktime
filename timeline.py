# gantt_by_procedure.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta, time
import itertools
import networkx as nx

st.set_page_config(page_title="Timeline theo Thủ thuật", layout="wide")
st.title("Timeline — mỗi dòng = 1 Thủ thuật, hiển thị BS / BN / Mã máy")

st.markdown("Upload file Excel hoặc nhập đường dẫn file local (app mặc định đã set theo file của anh)")

# -----------------------
# Upload or Load Excel
# -----------------------
uploaded = st.file_uploader("Chọn file Excel (.xlsx)", type=["xlsx"])
use_local = st.checkbox("Dùng file local", value=False)
local_path = st.text_input("Đường dẫn file local", "") if use_local else ""

df = None
if uploaded:
    df = pd.read_excel(uploaded)
elif use_local and local_path.strip():
    try:
        df = pd.read_excel(local_path)
    except Exception as e:
        st.error(f"Lỗi đọc file local: {e}")
        st.stop()
else:
    st.info("Chưa có file — hãy upload hoặc nhập path.")
    st.stop()

st.success(f"Đã load file — {df.shape[0]} dòng, {df.shape[1]} cột")

st.dataframe(df.head(8))

# -----------------------
# AUTO-DETECT TYPE (PTTT / CLS)
# -----------------------

headers = set(df.columns.str.upper())

# Điều kiện nhận dạng
pttt_signatures = {"NGAYPHAUTHUATTHUTHUAT", "KETTHUCPTTT", "PHAUTHUATVIEN"}
cls_signatures  = {"THOIGIANBATDAU", "THOIGIANKETTHUC", "BACSI"}

has_pttt = bool(pttt_signatures & headers)
has_cls  = bool(cls_signatures & headers)

if has_pttt and not has_cls:
    detected_type = "PTTT"
elif has_cls and not has_pttt:
    detected_type = "CLS"
else:
    # Nếu ambiguous → hỏi người dùng
    detected_type = st.radio(
        "Không xác định được kiểu file — chọn đúng loại dữ liệu:",
        ["PTTT", "CLS"],
        horizontal=True
    )

st.success(f"Đang xử lý dạng file: **{detected_type}**")

# -----------------------
# Default mapping
# -----------------------

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

elif detected_type == "CLS":
    default_map = {
        "col_patient": "MAHOSOBENHAN",
        "col_proc": "TENDICHVU",
        "col_dept": "PHONG",
        "col_start": "TGTHUCHIEN",
        "col_end": "TGKETQUA",
        "col_doc": "BACSITHUCHIEN",
        "col_machine": "MAMAY"
    }

# -----------------------
# Allow override (user adjust)
# -----------------------

st.markdown("### 🔧 Điều chỉnh tên cột (nếu header khác)")

col_patient = st.text_input("Cột Mã bệnh án", default_map["col_patient"])
col_proc    = st.text_input("Cột Tên dịch vụ", default_map["col_proc"])
col_dept    = st.text_input("Cột Khoa/Phòng", default_map["col_dept"])
col_start   = st.text_input("Cột Thời gian bắt đầu", default_map["col_start"])
col_end     = st.text_input("Cột Thời gian kết thúc", default_map["col_end"])
col_doc     = st.text_input("Cột Bác sĩ", default_map["col_doc"])
col_machine = st.text_input("Cột Mã máy", default_map["col_machine"])

required_columns = [
    col_patient, col_proc, col_dept,
    col_start, col_end, col_doc, col_machine
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"❌ Không tìm thấy các cột: {missing}")
    st.stop()


# -----------------------
# Parse datetimes
# -----------------------
def parse_dt(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.to_datetime(x)
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except:
            pass
    return pd.to_datetime(s, errors='coerce')

with st.spinner("Parsing datetime..."):
    df["_start"] = df[col_start].apply(parse_dt)
    df["_end"] = df[col_end].apply(parse_dt)

bad = df[df["_start"].isna() | df["_end"].isna()]
if not bad.empty:
    st.warning(f"{len(bad)} rows không parse được thời gian — xem sample", icon="⚠️")
    st.dataframe(bad.head(6))

# normalize
df["_proc"] = df[col_proc].astype(str)
df["_dept"] = df[col_dept].astype(str)
df["_doctor"] = df[col_doc].astype(str)
df["_patient"] = df[col_patient].astype(str)
def machine_label(v):
    s = str(v).strip()
    if s in ("-1","nan","", "None"):
        return "Không dùng máy"
    return s
df["_machine"] = df[col_machine].apply(machine_label)

# -----------------------
# Sidebar filters + work window for utilization
# -----------------------
st.sidebar.header("Bộ lọc")
dept_list = sorted(df["_dept"].dropna().unique())
sel_dept = st.sidebar.selectbox("Chọn Khoa/Phòng", ["(Tất cả)"] + dept_list)

date_min = df["_start"].min().date()
date_max = df["_end"].max().date()

sel_date = st.sidebar.date_input("Chọn ngày", value=date_min, min_value=date_min, max_value=date_max)
# time window for view (initial 7:00-12:00, rangeslider will allow moving)
view_start_default = datetime(sel_date.year, sel_date.month, sel_date.day, 7,0)
view_end_default = datetime(sel_date.year, sel_date.month, sel_date.day, 23,59)
time_window = st.sidebar.slider(
    "Khoảng giờ (kéo để phóng/thu, hoặc dùng rangeslider dưới chart)",
    value=(view_start_default, view_end_default),
    min_value=datetime(date_min.year, date_min.month, date_min.day, 0,0),
    max_value=datetime(date_max.year, date_max.month, date_max.day, 23,59),
    format="YYYY-MM-DD HH:mm"
)

# Working hours for utilization calc
st.sidebar.markdown("---")
work_start = st.sidebar.time_input("Bắt đầu ngày làm việc (util calc)", value=time(7,0))
work_end   = st.sidebar.time_input("Kết thúc ngày làm việc (util calc)", value=time(17,0))

# Filtering rows by dept and selected date and view window
df_f = df.copy()
if sel_dept != "(Tất cả)":
    df_f = df_f[df_f["_dept"] == sel_dept]

# keep rows that have any overlap with the chosen date
df_f = df_f[ (df_f["_start"].dt.date == sel_date) | (df_f["_end"].dt.date == sel_date) ]
# further filter by view window overlap
start_window, end_window = time_window
df_f = df_f[ ~(df_f["_end"] < start_window) & ~(df_f["_start"] > end_window) ]

st.write(f"Sau lọc có {len(df_f)} ca (ngày {sel_date})")

if df_f.empty:
    st.info("Không có ca nào trong bộ lọc này.")
    st.stop()

# Build plot_df
plot_df = df_f[["_proc","_start","_end","_doctor","_patient","_machine","_dept"]].copy()
plot_df = plot_df.rename(columns={
    "_proc":"Procedure",
    "_start":"Start",
    "_end":"End",
    "_doctor":"Doctor",
    "_patient":"Patient",
    "_machine":"Machine",
    "_dept":"Dept"
})
plot_df = plot_df.reset_index(drop=True)
plot_df["Label"] = plot_df["Doctor"] + " — " + plot_df["Patient"] + " — " + plot_df["Machine"]
# unique id for grouping
plot_df["id"] = plot_df.index.astype(str)

# -----------------------
# Conflict detection per pair and build graph for clusters
# -----------------------
def overlap(a_start, a_end, b_start, b_end):
    return a_start < b_end and a_end > b_start

# build edge list with reason tags
edges = []
for i, r in plot_df.iterrows():
    for j, s in plot_df.iterrows():
        if i >= j:
            continue
        if overlap(r["Start"], r["End"], s["Start"], s["End"]):
            reasons = []
            if r["Doctor"] == s["Doctor"] and r["Doctor"] != "":
                reasons.append("BS")
            if r["Patient"] == s["Patient"] and r["Patient"] != "":
                reasons.append("BN")
            if r["Machine"] == s["Machine"] and r["Machine"] != "Không dùng máy":
                reasons.append("M")
            if reasons:
                edges.append((str(i), str(j), {"reasons": reasons}))

G = nx.Graph()
G.add_nodes_from(plot_df["id"].tolist())
for a,b,attr in edges:
    G.add_edge(a,b, **attr)

# connected components -> groups
groups = list(nx.connected_components(G))
# remove singletons (no conflict)
conflict_groups = [g for g in groups if len(g) > 1]

# assign group ids and dominant reason
group_map = {}
group_info = []
for gi, g in enumerate(conflict_groups, start=1):
    members = sorted(list(g), key=lambda x: int(x))
    # collect reasons across edges inside group
    reason_counts = {"BS":0,"BN":0,"M":0}
    for a,b in itertools.combinations(members,2):
        if G.has_edge(a,b):
            for r in G[a][b]["reasons"]:
                reason_counts[r]+=1
    # dominant reason by count (tie-breaker order BS>BN>M)
    dominant = max(["BS","BN","M"], key=lambda k:(reason_counts[k], {"BS":3,"BN":2,"M":1}[k]))
    gid = f"Group {gi}"
    for m in members:
        group_map[m] = {"group": gid, "dominant": dominant}
    group_info.append({"group": gid, "members": members, "dominant": dominant, "reason_counts": reason_counts})

# populate plot_df conflict info
plot_df["ConflictType"] = "OK"
plot_df["ConflictGroup"] = ""
for idx in plot_df["id"]:
    if idx in group_map:
        dom = group_map[idx]["dominant"]
        plot_df.loc[plot_df["id"]==idx, "ConflictType"] = dom
        plot_df.loc[plot_df["id"]==idx, "ConflictGroup"] = group_map[idx]["group"]

# color map by ConflictType (priority BS>BN>M>OK)
COLOR_MAP = {"OK":"#2E86DE","BS":"#FF4D4D","BN":"#FFA500","M":"#9D4EDD"}
plot_df["Color"] = plot_df["ConflictType"].map(lambda t: COLOR_MAP.get(t,"#2E86DE"))

# -----------------------
# Machine 'hố nhau' (explicit machine overlaps) and downtime / usage calc
# -----------------------
# We'll compute per machine:
# - total busy minutes in the selected work window (work_start-work_end)
# - gaps between consecutive bookings
# - downtime percentage (gap / work_window_minutes)
# - heatmap per hour (count minutes used per hour bucket)

work_day_start = datetime.combine(sel_date, work_start)
work_day_end = datetime.combine(sel_date, work_end)
work_window_minutes = int((work_day_end - work_day_start).total_seconds()/60)
machines = sorted(plot_df["Machine"].unique())

usage_stats = []
heat_rows = []
for m in machines:
    rows = plot_df[plot_df["Machine"]==m].sort_values("Start")
    busy_minutes = 0
    gaps = []
    last_end = None
    # Build per-minute occupancy array for 24 hours (or work window)
    minutes = np.zeros(24*60, dtype=int)  # index 0 = 00:00
    for _, r in rows.iterrows():
        s = max(r["Start"], datetime.combine(sel_date, time(0,0)))
        e = min(r["End"], datetime.combine(sel_date, time(23,59)))
        if s >= e:
            continue
        duration_min = int((e - s).total_seconds() / 60)
        busy_minutes += duration_min
        # fill minutes
        start_idx = s.hour*60 + s.minute
        end_idx = e.hour*60 + e.minute
        minutes[start_idx:end_idx] += 1
        # gaps
        if last_end is not None and s > last_end:
            gap_min = int((s - last_end).total_seconds()/60)
            gaps.append(gap_min)
        last_end = max(last_end, e) if last_end is not None else e
    # compute usage within work window
    s_win_idx = work_day_start.hour*60 + work_day_start.minute
    e_win_idx = work_day_end.hour*60 + work_day_end.minute
    busy_in_window = minutes[s_win_idx:e_win_idx].sum()
    util_pct = (busy_in_window / max(1, (e_win_idx - s_win_idx))) * 100
    avg_gap = int(np.mean(gaps)) if gaps else 0
    usage_stats.append({
        "machine": m,
        "num_cases": len(rows),
        "busy_minutes": int(busy_minutes),
        "avg_gap_min": avg_gap,
        "downtime_pct": round((1 - util_pct/100)*100,2),
        "util_pct": round(util_pct,2)
    })
    # build hourly counts for heatmap
    hourly = [minutes[h*60:(h+1)*60].sum() for h in range(24)]
    heat_rows.append(hourly)

heat_df = pd.DataFrame(heat_rows, index=machines, columns=[f"{h:02d}:00" for h in range(24)])

# -----------------------
# Display tables & group info
# -----------------------
st.subheader("⚠️ Nhóm xung đột (Clustered conflicts)")
if not group_info:
    st.success("Không phát hiện nhóm xung đột nào — ok ạ 😎")
else:
    for g in group_info:
        st.markdown(f"**{g['group']}** — Lý do chính: **{g['dominant']}** — Thành viên: {len(g['members'])}")
        for m in g['members']:
            idx = int(m)
            r = plot_df.loc[idx]
            st.markdown(f"- Ca `{m}`: {r['Procedure']} | BS: **{r['Doctor']}** | BN: **{r['Patient']}** | Máy: **{r['Machine']}** | {r['Start'].strftime('%H:%M')}–{r['End'].strftime('%H:%M')}")

st.subheader("📊 Thống kê sử dụng máy")
st.dataframe(pd.DataFrame(usage_stats).sort_values("util_pct", ascending=False).reset_index(drop=True))

st.subheader("🔸 Heatmap sử dụng theo giờ (số phút hoạt động/giờ)")
fig_heat = px.imshow(heat_df.values,
                    labels=dict(x="Giờ trong ngày", y="Máy", color="Số phút hoạt động"),
                    x=heat_df.columns, y=heat_df.index,
                    aspect="auto",
                    color_continuous_scale="YlOrRd")
st.plotly_chart(fig_heat, use_container_width=True, key="heatmap")

# -----------------------
# Prepare Tooltip & Timeline plot
# -----------------------
plot_df["Tooltip"] = (
    "⏱ " + plot_df["Start"].dt.strftime("%H:%M") +
    " - " + plot_df["End"].dt.strftime("%H:%M") +
    "<br>🧪 " + plot_df["Procedure"] +
    "<br>🩺 BS: " + plot_df["Doctor"] +
    "<br>📄 BA: " + plot_df["Patient"] +
    "<br>🔧 Máy: " + plot_df["Machine"] +
    plot_df["ConflictType"].apply(lambda x: "<br>⚠️ Trùng: " + x if x != "OK" else "")
)

fig = px.timeline(
    plot_df,
    x_start="Start",
    x_end="End",
    y="Procedure",
    color="Color",
    color_discrete_map="identity",
    custom_data=["Tooltip","ConflictGroup","id"]
)

# hovertemplate using customdata[0] (Tooltip)
fig.update_traces(hovertemplate="%{customdata[0]}")

fig.update_yaxes(autorange="reversed")
fig.update_layout(
    height=780,
    hoverlabel=dict(font_size=16),
    xaxis=dict(
        range=[time_window[0], time_window[1]],
        rangeslider=dict(visible=True),
        type="date"
    ),
    legend_title_text="Trạng thái"
)

st.subheader("Timeline theo Thủ thuật")
st.plotly_chart(fig, use_container_width=True, key="timeline_main")

# -----------------------
# Final: export CSV of conflicts/groups if needed
# -----------------------
export_df = plot_df[["id","Procedure","Doctor","Patient","Machine","Start","End","ConflictType","ConflictGroup"]]
csv = export_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("Tải CSV báo cáo xung đột", data=csv, file_name="conflicts_report.csv", mime="text/csv")
