def rule_based_spam_filter(text, threshold):
    text_no_space = text.replace(' ', '')
    total_chars = len(text_no_space)
    if total_chars == 0:
        return False, None 
    
    non_letter_count = sum(1 for char in text_no_space if not char.isalpha())
    proportion = non_letter_count / total_chars
    
    if proportion >= threshold:
        return True, "Terlalu banyak simbol"  
    else:
        return False, None  