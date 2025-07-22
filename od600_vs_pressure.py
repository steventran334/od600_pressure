import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import h5py
import scipy.io

def get_matlab_struct_field(struct, field):
    # Extract MATLAB struct field from scipy.io.loadmat output
    val = struct[0,0][field][0,0]
    if hasattr(val, 'shape') and val.shape == (1, 1):
        return val[0,0]
    return val

def get_row(arr, idx):
    arr = np.asarray(arr)
    if arr.ndim == 0:        # scalar
        return np.array([arr])
    if arr.ndim == 1:
        return arr
    return arr[idx, :]

st.title("OD600 vs Pressure Plotter (.mat files)")

uploaded_file = st.file_uploader("Upload your .mat file", type=["mat"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Try loading as HDF5 first (MATLAB v7.3+), then fallback to scipy
    try:
        with h5py.File(tmp_file_path, 'r') as f:
            def arr(dataset):
                data = np.array(dataset)
                if data.ndim == 2 and data.shape[0] == 1:
                    data = data[0]
                elif data.ndim == 2 and data.shape[1] == 1:
                    data = data[:,0]
                return data
            spectdata = f['spectdata']
            wavelengths = arr(spectdata['wavelengths'])
            sample = np.array(spectdata['sample'])
            reference = np.array(spectdata['reference'])
            pressuremeasured = arr(spectdata['pressuremeasured'])
            has_dark = 'dark' in spectdata
            if has_dark:
                dark = np.array(spectdata['dark'])
    except Exception:
        mat = scipy.io.loadmat(tmp_file_path)
        spectdata = mat['spectdata']
        wavelengths = get_matlab_struct_field(spectdata, 'wavelengths').flatten()
        sample = get_matlab_struct_field(spectdata, 'sample')
        reference = get_matlab_struct_field(spectdata, 'reference')
        pressuremeasured = get_matlab_struct_field(spectdata, 'pressuremeasured').flatten()
        try:
            dark = get_matlab_struct_field(spectdata, 'dark')
            has_dark = True
        except Exception:
            has_dark = False

    idx600 = np.argmin(np.abs(wavelengths - 600))

    if has_dark:
        I_dark = get_row(dark, idx600)
        I_sample = get_row(sample, idx600) - I_dark
        I_ref = get_row(reference, idx600) - I_dark
    else:
        I_sample = get_row(sample, idx600)
        I_ref = get_row(reference, idx600)
    od600 = -np.log10(I_sample / I_ref)

    fig, ax = plt.subplots()
    ax.plot(pressuremeasured, od600, 'o-')
    ax.set_xlabel('Measured Pressure (kPa)')
    ax.set_ylabel('OD$_{600}$')
    ax.set_title('OD$_{600}$ vs Pressure')
    ax.grid(True)
    st.pyplot(fig)
