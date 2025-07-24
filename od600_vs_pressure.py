import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import scipy.io
import io
import os

st.title("OD600 vs Pressure Plotter (multiple .mat overlay)")

uploaded_files = st.file_uploader(
    "Upload one or more .mat files", 
    type=["mat"], 
    accept_multiple_files=True
)
plot_title = st.text_input("Plot Title", value="OD600 vs Pressure")

if uploaded_files:
    fig, ax = plt.subplots()
    colors = plt.cm.tab10.colors  # for up to 10 curves

    for idx, uploaded_file in enumerate(uploaded_files):
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load MATLAB struct (v7.2 or lower)
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

        # Use filename (without extension/path) as label
        label = os.path.splitext(uploaded_file.name)[0]
        color = colors[idx % len(colors)]
        ax.plot(
            pressuremeasured, od600, 'o-', 
            label=label, color=color
        )

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
