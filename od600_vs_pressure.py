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
plot_title = st.text_input("Plot Title", value="OD600 vs Pressure")

series_names = []
if uploaded_files:
    # Input for custom series names
    st.subheader("Edit Series Names")
    for idx, f in enumerate(uploaded_files):
        default = os.path.splitext(f.name)[0]
        name = st.text_input(f"Series {idx+1} name:", value=default, key=f"seriesname{idx}")
        series_names.append(name)

    fig, ax = plt.subplots()
    colors = plt.cm.tab10.colors
    results = []  # To store summary for table

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

        # Ensure arrays are sorted in ascending pressure for correct interpolation
        if not np.all(np.diff(pressuremeasured) > 0):
            sort_idx = np.argsort(pressuremeasured)
            pressuremeasured = pressuremeasured[sort_idx]
            od600 = od600[sort_idx]

        # Plot with custom label
        label = series_names[idx]
        color = colors[idx % len(colors)]
        ax.plot(
            pressuremeasured, od600, 'o-', 
            label=label, color=color
        )

        # Calculate OD600 thresholds: 100%, 75%, 50%, 25%, 0%
        max_val = od600[0]  # 100% at lowest pressure
        min_val = od600[-1] # 0% at highest pressure
        thresholds = [1.0, 0.75, 0.5, 0.25, 0.0]  # 100%, 75%, 50%, 25%, 0%
        level_dict = {}
        for thresh in thresholds:
            target = min_val + (max_val - min_val) * thresh
            idx_nearest = np.argmin(np.abs(od600 - target))
            p_val = pressuremeasured[idx_nearest]
            level_dict[f"{int(thresh*100)}% OD600"] = (target, p_val)

        # Store for summary table
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

    # Show summary table
    st.subheader("Pressure at OD600 transition points")
    df = pd.DataFrame(results)
    st.dataframe(df)
