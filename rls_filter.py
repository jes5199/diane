import numpy as np

def normalize_audio(signal):
    """Normalize audio to [-1, 1] range"""
    return signal / 32768.0

def denormalize_audio(signal):
    """Convert back to int16 range"""
    return np.clip(signal * 32768.0, -32768, 32767)

def rls_filter(desired, reference, filter_coeff, reg_param=0.1, forget_factor=0.999):
    """
    RLS (Recursive Least Squares) adaptive filter implementation.
    
    Args:
        desired: Target signal to be cleaned
        reference: Reference signal (contains echo)
        filter_coeff: Initial filter coefficients
        reg_param: Regularization parameter (delta)
        forget_factor: Forgetting factor (lambda) for giving less weight to old samples
    
    Returns:
        filtered: The processed signal
        error: Error signal for each sample
    """
    # Normalize inputs
    desired = normalize_audio(desired)
    reference = normalize_audio(reference)
    filter_coeff = normalize_audio(filter_coeff)
    
    filter_len = len(filter_coeff)
    signal_len = len(desired)
    
    # Initialize inverse correlation matrix
    P = np.eye(filter_len) / reg_param
    
    # Pre-allocate output arrays
    filtered = np.zeros(signal_len, dtype=np.float32)
    error = np.zeros(signal_len, dtype=np.float32)
    
    # Process each sample
    for n in range(filter_len, signal_len):
        # Get current chunk of reference signal
        ref_chunk = reference[n - filter_len:n]
        
        # Compute filter output
        filtered[n] = np.dot(filter_coeff, ref_chunk)
        
        # Calculate error
        error[n] = desired[n] - filtered[n]
        
        # Update P matrix
        k_den = forget_factor + np.dot(np.dot(ref_chunk, P), ref_chunk)
        if k_den > 1e-10:  # Numerical stability check
            # Compute Kalman gain
            k = np.dot(P, ref_chunk) / k_den
            
            # Update inverse correlation matrix
            P = (P - np.outer(k, np.dot(ref_chunk, P))) / forget_factor
            
            # Update filter coefficients
            filter_coeff = filter_coeff + k * error[n]
    
    return denormalize_audio(filtered), error

def rls_filter_safe(desired, reference, filter_coeff, reg_params=[0.1, 0.01, 0.001]):
    """
    RLS filter with automatic parameter selection.
    
    Args:
        desired: Target signal to be cleaned
        reference: Reference signal (contains echo)
        filter_coeff: Initial filter coefficients
        reg_params: List of regularization parameters to try
    """
    best_error = float('inf')
    best_filtered = None
    best_param = None
    
    for param in reg_params:
        # Apply filter with current regularization parameter
        filtered, error = rls_filter(desired, reference, filter_coeff, reg_param=param)
        
        # Calculate error metric
        error_metric = np.mean(np.abs(error))
        
        # Update best if this is better
        if error_metric < best_error:
            best_error = error_metric
            best_filtered = filtered
            best_param = param
    
    return best_filtered, best_param