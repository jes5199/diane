import numpy as np

def normalize_audio(signal):
    """Normalize audio to [-1, 1] range"""
    return signal / 32768.0  # int16 range is -32768 to 32767

def denormalize_audio(signal):
    """Convert back to int16 range"""
    return np.clip(signal * 32768.0, -32768, 32767)

def lms_filter_safe(desired, reference, filter_coeff, step_sizes):
    """
    LMS filter with automatic step size selection.
    
    Args:
        desired: Target signal to be cleaned (int16 range)
        reference: Reference signal (contains echo) (int16 range)
        filter_coeff: Initial filter coefficients
        step_sizes: List of step sizes to try
    """
    # Normalize inputs to [-1, 1] range
    desired_norm = normalize_audio(desired)
    reference_norm = normalize_audio(reference)
    filter_coeff = normalize_audio(filter_coeff)
    
    best_error = float('inf')
    best_filtered = None
    best_step = None
    
    for step in step_sizes:
        # Apply filter with current step size
        filtered, error = lms_filter(desired_norm, reference_norm, filter_coeff.copy(), step)
        
        # Calculate error metric (mean squared error)
        error_metric = np.mean(np.abs(error))  # Using absolute error instead of squared
        
        # Update best if this is better
        if error_metric < best_error:
            best_error = error_metric
            best_filtered = filtered
            best_step = step
    
    # Convert back to int16 range
    return denormalize_audio(best_filtered), best_step

def lms_filter(desired, reference, filter_coeff, step_size):
    """
    Basic LMS adaptive filter implementation.
    Assumes all inputs are already normalized to [-1, 1] range.
    """
    filter_len = len(filter_coeff)
    signal_len = len(desired)
    
    # Pre-allocate output arrays
    error = np.zeros(signal_len, dtype=np.float32)
    filtered = np.zeros(signal_len, dtype=np.float32)
    
    # Process each sample
    for n in range(filter_len, signal_len):
        # Get the current chunk of reference signal
        ref_chunk = reference[n - filter_len:n]
        
        # Calculate filter output
        filtered[n] = np.dot(filter_coeff, ref_chunk)
        
        # Calculate error
        error[n] = desired[n] - filtered[n]
        
        # Normalize step size by signal power to prevent overflow
        signal_power = np.dot(ref_chunk, ref_chunk) + 1e-10
        normalized_step = step_size / signal_power
        
        # Update filter coefficients
        filter_coeff = filter_coeff + normalized_step * error[n] * ref_chunk
        
        # Keep coefficients in reasonable range
        filter_coeff = np.clip(filter_coeff, -1, 1)
        
    return filtered, error