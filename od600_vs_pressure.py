import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import h5py
import scipy.io

def get_mat_array(dataset):
    data = np.array(dataset)
    if data.ndim == 2 and data.shape[0] == 1:
        data = data[0]
    elif data.ndim == 2 and data.shape[1] == 1:
        data = data[:,0]
    return data

st.title("OD600 vs Pressure Plotter (.mat files)")

uploaded_file = st.file_uploader("Upload your .mat file", type=["mat"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Try HDF5 loader first
    try:
        with h5py.File(tmp_file_path, 'r') as f:
            mat_type = "hdf5"
            wavelengths = get_mat_array(f['spectdata']['wavelengths'])
            sample = np.array(f['spectdata']['sample'])
            reference = np.array(f['spectdata']['reference'])
            dark = np.array(f['spectdata']['dark'])
            pressuremeasured = get_mat_array(f['spectdata']['pressuremeasured'])
    except Exception:
        # Try scipy for pre-v7.3 mat files
        try:
            mat = scipy.io.loadmat(tmp_file_path)
            mat_type = "scipy"
            # Field names may vary, update if needed
            spectdata = mat['spectdata']
            # You may need to check struct dtype in scipy
            def extract(field):
                # Handle MATLAB structs loaded as numpy voids
                val = spectdata[0,0][field][0,0]
                if val.shape == (1, 1):
                    return val[0,0]
                else:
                    return val
            wavelengths = extract('wavelengths').flatten()
            sample = extract('sample')
            reference = extract('reference')
            dark = extract('dark')
            pressuremeasured = extract('pressuremeasured').flatten()
        except Exception as e:
            st.error("This file is not a valid MATLAB .mat file, or the format is not supported. Error: " + str(e))
            st.stop()
    
    # Find index closest to 600 nm
    idx600 = np.argmin(np.abs(wavelengths - 600))
    I_dark = dark[idx600, :] if dark.shape[0] > 1 else dark[0, :]
    I_sample = sample[idx600, :] - I_dark
    I_ref = reference[idx600, :] - I_dark
    od600 = -np.log10(I_sample / I_ref)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(pressuremeasured, od600, 'o-')
    ax.set_xlabel('Measured Pressure (kPa)')
    ax.set_ylabel('OD600')
    ax.set_title('OD600 vs Pressure')
    ax.grid(True)
    st.pyplot(fig)
