def generate_unique_filename(prefix='file', extension='.pt'):
    timestamp = time.strftime('%Y%m%d%H%M%S')
    unique_filename = f"{prefix}_{timestamp}{extension}"
    return unique_filename