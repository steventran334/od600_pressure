import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import scipy.io

st.title("OD600 vs Pressure Plotter (.mat files)")

uploaded_file = st.file_uploader("Upload your .mat file", type=["mat"])
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

    fig, ax = plt.subplots()
    ax.plot(pressuremeasured, od600, 'o-')
    ax.set_xlabel('Measured Pressure (kPa)')
    ax.set_ylabel('OD$_{600}$')
    ax.set_title('OD$_{600}$ vs Pressure')
    ax.grid(True)
    st.pyplot(fig)
