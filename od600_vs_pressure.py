import streamlit as st
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tempfile

st.title("OD600 vs Pressure Plotter (.mat files)")

uploaded_file = st.file_uploader("Upload your .mat file", type=["mat"])
if uploaded_file:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    with h5py.File(tmp_file_path, 'r') as f:
        # Helper to extract dataset as numpy array and transpose if needed
        def get_mat_array(dataset):
            data = np.array(dataset)
            if data.ndim == 2 and data.shape[0] == 1:
                data = data[0]
            elif data.ndim == 2 and data.shape[1] == 1:
                data = data[:,0]
            return data
        
        wavelengths = get_mat_array(f['spectdata']['wavelengths'])
        sample = np.array(f['spectdata']['sample'])
        reference = np.array(f['spectdata']['reference'])
        dark = np.array(f['spectdata']['dark'])
        pressuremeasured = get_mat_array(f['spectdata']['pressuremeasured'])

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
