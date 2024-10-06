import ptufile
import numpy as np
import lmfit
import napari

def load_ptu_file(filepath):
    """Load a PTU file using ptufile and decode the photon data."""
    with ptufile.PtuFile(filepath) as ptu:
        # Example: Decode time-correlated single photon counting (TCSPC) records
        records = ptu.decode_records()
        return records
    

def n_exponential_tail_fit(time, intensity, n):
    """Fit the intensity decay curve with an n-exponential tail fit model."""
    def model(params, time):
        decay = np.zeros_like(time)
        for i in range(n):
            A = params[f'A{i+1}']
            tau = params[f'tau{i+1}']
            decay += A * np.exp(-time / tau)
        return decay

    def residual(params, time, intensity):
        return intensity - model(params, time)

    params = lmfit.Parameters()
    for i in range(n):
        params.add(f'A{i+1}', value=1, min=0)
        params.add(f'tau{i+1}', value=1, min=0)

    result = lmfit.minimize(residual, params, args=(time, intensity))
    return result.params


def calculate_tau_average(params, n):
    """Calculate the average lifetime from the fitted parameters."""
    tau_avg = sum(params[f'A{i+1}'].value * params[f'tau{i+1}'].value for i in range(n))
    total_A = sum(params[f'A{i+1}'].value for i in range(n))
    return tau_avg / total_A


def decode_flim_image(filepath):
    """Decode FLIM image data from the PTU file."""
    with ptufile.PtuFile(filepath) as ptu:
        # Decode image histogram data (for FLIM)
        image_data = ptu.decode_image(channel=0, dtime=-1, asxarray=True)
        return image_data
    

def display_flim_image(image_data):
    """Display the FLIM image in Napari and allow the user to define ROIs."""
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image_data, name='FLIM Image')
        roi_layer = viewer.add_shapes(name='ROIs', shape_type='polygon')  # Users can define ROIs here
        return viewer, roi_layer

def get_region_average_lifetime(roi, image_data, time_data, n):
    """Calculate average lifetime for a specific ROI."""
    mask = np.zeros_like(image_data)
    rr, cc = np.round(roi[:, 0]).astype(int), np.round(roi[:, 1]).astype(int)
    mask[rr, cc] = 1
    region_intensity = np.mean(image_data[mask == 1], axis=0)
    fitted_params = n_exponential_tail_fit(time_data, region_intensity, n)
    return calculate_tau_average(fitted_params, n)
