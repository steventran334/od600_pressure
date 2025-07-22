import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import scipy.io
import io

st.title("OD600 vs Pressure Plotter (.mat files)")

uploaded_file = st.file_uploader("Upload your .mat file", type=["mat"])
plot_title = st.text_input("Plot Title", value="OD600 vs Pressure")

if uploaded_file:
    # Save to a temporary file
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

    # Find index for 600 nm
    idx600 = np.argmin(np.abs(wavelengths - 600))

    if has_dark:
        I_dark = dark[idx600, :]
        I_sample = sample[idx600, :] - I_dark
        I_ref = reference[idx600, :] - I_dark
    else:
        I_sample = sample[idx600, :]
        I_ref = reference[idx600, :]

    od600 = -np.log10(I_sample / I_ref)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(pressuremeasured, od600, 'o-')
    ax.set_xlabel('Measured Pressure (kPa)')
    ax.set_ylabel('OD$_{600}$')
    ax.set_title(plot_title)
    ax.grid(True)
    st.pyplot(fig)

    # Download SVG button
    svg_buf = io.StringIO()
    fig.savefig(svg_buf, format="svg")
    svg_data = svg_buf.getvalue()
    st.download_button(
        label="Download plot as SVG",
        data=svg_data,
        file_name="OD600_vs_Pressure.svg",
        mime="image/svg+xml"
    )
