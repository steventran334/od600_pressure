import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import scipy.io
import io
import os
import pandas as pd

st.title("OD600 vs Pressure Plotter (multiple .mat overlay)")

uploaded_files = st.file_uploader(
    "Upload one or more .mat files", 
    type=["mat"], 
    accept_multiple_files=True
)

# Editable title for the absolute (raw) overlay plot
plot_title = st.text_input("Plot Title (for absolute OD600 plot)", value="OD600 vs Pressure")

series_names = []
od600_all = []
pressure_all = []
max_od600s = []

if uploaded_files:
    st.subheader("Edit Series Names")
    for idx, f in enumerate(uploaded_files):
        default = os.path.splitext(f.name)[0]
        name = st.text_input(f"Series {idx+1} name:", value=default, key=f"seriesname{idx}")
        series_names.append(name)

    fig, ax = plt.subplots()
    colors = plt.cm.tab10.colors
    results = []  # For percent summary table

    for idx, uploaded_file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        mat = scipy.io.loadmat(tmp_file_path)
        spectdata = mat['spectdata'][0,0]
        wavelengths = spectdata['wavelengths'].squeeze()
        sample = spectdata['sample']
        reference = spectdata['reference']
        pressuremeasured = spectdata['pressuremeasured'].squeeze()
        try:
            dark = spectdata['dark']
            has_dark = True
        except Exception:
            has_dark = False

        idx600 = np.argmin(np.abs(wavelengths - 600))
        if has_dark:
            I_dark = dark[idx600, :]
            I_sample = sample[idx600, :] - I_dark
            I_ref = reference[idx600, :] - I_dark
        else:
            I_sample = sample[idx600, :]
            I_ref = reference[idx600, :]

        od600 = -np.log10(I_sample / I_ref)

        # Ensure ascending order in pressure for fair comparison/interpolation
        if not np.all(np.diff(pressuremeasured) > 0):
            sort_idx = np.argsort(pressuremeasured)
            pressuremeasured = pressuremeasured[sort_idx]
            od600 = od600[sort_idx]

        # For later plots and tables
        od600_all.append(od600)
        pressure_all.append(pressuremeasured)
        max_od600 = np.max(od600)
        max_od600s.append(max_od600)

        # Main (absolute) overlay plot
        label = series_names[idx]
        color = colors[idx % len(colors)]
        ax.plot(
            pressuremeasured, od600, 'o-',
            label=label, color=color
        )

        # Percent threshold summary (relative to max OD600 at lowest pressure)
        max_val = od600[0]  # 100% at lowest pressure
        thresholds = [1.0, 0.75, 0.5, 0.25, 0.0]
        level_dict = {}
        for thresh in thresholds:
            target = max_val * thresh
            idx_nearest = np.argmin(np.abs(od600 - target))
            p_val = pressuremeasured[idx_nearest]
            level_dict[f"{int(thresh*100)}% OD600"] = (target, p_val)
        row = {"Series": label}
        for key in level_dict:
            row[key] = level_dict[key][1]
        results.append(row)

    ax.set_xlabel('Measured Pressure (kPa)')
    ax.set_ylabel('OD$_{600}$')
    ax.set_title(plot_title)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Download SVG button (for overlay plot)
    svg_buf = io.StringIO()
    fig.savefig(svg_buf, format="svg")
    svg_data = svg_buf.getvalue()
    st.download_button(
        label="Download plot as SVG",
        data=svg_data,
        file_name="OD600_vs_Pressure_overlay.svg",
        mime="image/svg+xml"
    )

    # --- CSV EXPORT SECTION ---
    # Prepare data for CSV export
    export_rows = []
    for idx, (series, pressure, od600) in enumerate(zip(series_names, pressure_all, od600_all)):
        for p, od in zip(pressure, od600):
            export_rows.append({
                "Series": series,
                "Pressure (kPa)": p,
                "OD600": od
            })
    export_df = pd.DataFrame(export_rows)

    st.subheader("Download OD600 vs Pressure Data")
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="Download OD600 vs Pressure CSV",
        data=csv,
        file_name="OD600_vs_Pressure.csv",
        mime="text/csv"
    )
    # --- END CSV EXPORT SECTION ---

    # Show summary table (percent transitions)
    st.subheader("Pressure at OD600 transition points\n(Percent of max OD600 at lowest pressure)")
    df = pd.DataFrame(results)
    st.dataframe(df)

    # Max OD600 table
    st.subheader("Maximum OD600 for each series")
    max_df = pd.DataFrame({
        "Series": series_names,
        "Max OD600": max_od600s
    })
    st.dataframe(max_df)

    # Editable title for normalized overlay plot
    norm_plot_title = st.text_input("Normalized Plot Title", value="Normalized OD600 vs Pressure")
    st.subheader("Normalized OD600 vs Pressure (for fair comparison)")
    fig2, ax2 = plt.subplots()
    for idx, (pressuremeasured, od600) in enumerate(zip(pressure_all, od600_all)):
        max_val = od600[0]
        min_val = od600[-1]
        denom = max_val - min_val if (max_val - min_val) != 0 else 1.0
        norm_od600 = (od600 - min_val) / denom
        label = series_names[idx]
        color = colors[idx % len(colors)]
        ax2.plot(pressuremeasured, norm_od600, 'o-', label=label, color=color)
    ax2.set_xlabel('Measured Pressure (kPa)')
    ax2.set_ylabel('Normalized OD$_{600}$')
    ax2.set_title(norm_plot_title)
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    # Download SVG button for normalized plot
    svg_buf2 = io.StringIO()
    fig2.savefig(svg_buf2, format="svg")
    svg_data2 = svg_buf2.getvalue()
    st.download_button(
        label="Download normalized plot as SVG",
        data=svg_data2,
        file_name="Normalized_OD600_vs_Pressure_overlay.svg",
        mime="image/svg+xml"
    )
