import numpy as np

def normalize_audio(signal):
    """Normalize audio to [-1, 1] range"""
    return signal / 32768.0

def denormalize_audio(signal):
    """Convert back to int16 range"""
    return np.clip(signal * 32768.0, -32768, 32767)

class HybridFilter:
    def __init__(self, filter_length=96):
        self.filter_length = filter_length
        
        # RLS parameters
        self.rls_coeff = np.random.randn(filter_length)
        self.P = np.eye(filter_length) / 0.1
        self.forget_factor = 0.95
        
        # NLMS parameters
        self.nlms_coeff = np.random.randn(filter_length)
        self.step_size = 0.1
        self.eps = 1e-6
        
        # Double-talk detection
        self.dtd_threshold = 1.5
        self.hangover = 0
        self.hangover_period = 32  # samples
        
    def reset(self):
        """Reset filter states"""
        self.rls_coeff = np.random.randn(self.filter_length)
        self.nlms_coeff = np.random.randn(self.filter_length)
        self.P = np.eye(self.filter_length) / 0.1
        self.hangover = 0
        
    def detect_double_talk(self, input_chunk, reference_chunk):
        """
        Detect when both voice and echo are present
        Returns True if double-talk is detected
        """
        input_energy = np.mean(input_chunk ** 2)
        ref_energy = np.mean(reference_chunk ** 2)
        
        # Check if input energy is significantly higher than reference
        ratio = input_energy / (ref_energy + self.eps)
        
        is_double_talk = ratio > self.dtd_threshold
        
        # Add hangover time to prevent rapid switching
        if is_double_talk:
            self.hangover = self.hangover_period
        elif self.hangover > 0:
            is_double_talk = True
            self.hangover -= 1
            
        return is_double_talk
        
    def process(self, input_chunk, reference_chunk):
        """
        Process audio using hybrid RLS-NLMS approach
        """
        # Normalize input
        input_norm = normalize_audio(input_chunk)
        ref_norm = normalize_audio(reference_chunk)
        
        # Detect double-talk
        is_double_talk = self.detect_double_talk(input_norm, ref_norm)
        
        # If double-talk detected, reduce echo cancellation strength
        if is_double_talk:
            ref_norm *= 0.3
        
        # Apply RLS filter
        rls_out = np.zeros_like(input_norm)
        for n in range(self.filter_length, len(input_norm)):
            ref_window = ref_norm[n - self.filter_length:n]
            
            # RLS output
            rls_out[n] = np.dot(self.rls_coeff, ref_window)
            
            # RLS update
            error = input_norm[n] - rls_out[n]
            k_den = self.forget_factor + np.dot(np.dot(ref_window, self.P), ref_window)
            if k_den > 1e-10:
                k = np.dot(self.P, ref_window) / k_den
                self.P = (self.P - np.outer(k, np.dot(ref_window, self.P))) / self.forget_factor
                self.rls_coeff = self.rls_coeff + k * error
        
        # Apply NLMS filter
        nlms_out = np.zeros_like(input_norm)
        for n in range(self.filter_length, len(input_norm)):
            ref_window = ref_norm[n - self.filter_length:n]
            
            # NLMS output
            nlms_out[n] = np.dot(self.nlms_coeff, ref_window)
            
            # NLMS update
            error = input_norm[n] - nlms_out[n]
            norm = np.dot(ref_window, ref_window) + self.eps
            self.nlms_coeff += self.step_size * error * ref_window / norm
        
        # Combine outputs based on double-talk detection
        if is_double_talk:
            # During double-talk, favor NLMS as it's more stable
            output = 0.7 * nlms_out + 0.3 * rls_out
        else:
            # Otherwise use more RLS as it's more aggressive
            output = 0.3 * nlms_out + 0.7 * rls_out
            
        return denormalize_audio(output)