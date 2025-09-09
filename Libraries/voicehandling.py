import re
import unicodedata

def clean_function_text(func_text):
    """
    Clean a function definition by:
    1. Removing MCP calls like ⚡️<any_text>({}) - matches lightning bolt + any text + ({...})
    2. Removing emojis
    3. Adding commas to abbreviations (AM -> A, M, NYPD -> N, Y, P, D)
    4. Removing parentheses and quotation marks
    5. Converting symbols to words: = to equals, % to percent, # to hashtag, @ to at, & to and, / to slash, ² to squared, ³ to cubed, + to plus, - to minus, $ to dollar, etc.
    
    Args:
        func_text (str): The function text to clean
        
    Returns:
        str: The cleaned function text
    """
    
    # Step 1: Remove MCP calls using regex
    # This pattern matches ⚡️<any_text>({...}) with any content inside the braces
    mcp_pattern = r'⚡️[^(]*\(\{[^}]*\}\)'
    cleaned_text = re.sub(mcp_pattern, '', func_text)
    
    # Step 2: Remove emojis
    cleaned_text = remove_emojis(cleaned_text)
    
    # Step 3: Add commas to abbreviations
    cleaned_text = add_commas_to_abbreviations(cleaned_text)
    
    # Step 4: Apply additional symbol replacements
    cleaned_text = replace_symbols(cleaned_text)
    
    # Clean up extra whitespace that might have been left behind
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def remove_emojis(text):
    """
    Remove emojis from text using Unicode categories
    """
    # Remove emojis by filtering out characters in emoji-related Unicode categories
    cleaned = []
    for char in text:
        # Get the Unicode category of the character
        cat = unicodedata.category(char)
        # Skip emoji and symbol characters
        if not (cat.startswith('So') or  # Other symbols (includes most emojis)
                cat.startswith('Sm') or  # Math symbols
                cat.startswith('Sc') or  # Currency symbols
                unicodedata.name(char, '').startswith('EMOJI')):
            cleaned.append(char)
    
    return ''.join(cleaned)

def replace_symbols(text):
    """
    Replace various symbols and punctuation with their word equivalents
    """
    replacements = {
        # Basic punctuation and symbols
        '(': '',          # Remove opening parentheses
        ')': '',          # Remove closing parentheses
        '¡': '',          # Remove Exclamations
        '!': '. ',     # Replace ! with period and space
        '"': '',          # Remove double quotes
        "'": '',          # Remove single quotes
        '=': ' equals ',  # Replace = with equals
        '%': ' percent ', # Replace % with percent
        '#': ' hashtag ', # Replace # with hashtag
        '@': ' at ',      # Replace @ with at
        '&': ' and ',     # Replace & with and
        '/': ' slash ',   # Replace / with slash

        # Words
        'Hola': 'oh lah',
        'hola': 'oh lah',

        
        # Mathematical symbols
        '+': ' plus ',    # Addition
        '-': ' minus ',   # Subtraction  
        '*': ' times ',   # Multiplication
        '÷': ' divided by ', # Division symbol
        '×': ' times ',   # Multiplication symbol
        '±': ' plus or minus ', # Plus-minus
        '∞': ' infinity ', # Infinity
        '√': ' square root ', # Square root
        '∑': ' sum ',     # Summation
        '∏': ' product ', # Product
        '∆': ' delta ',   # Delta
        'π': ' pi ',      # Pi
        
        # Superscripts for powers
        '²': ' squared ', # Squared
        '³': ' cubed ',   # Cubed
        '⁴': ' to the fourth power ',
        '⁵': ' to the fifth power ',
        '⁶': ' to the sixth power ',
        '⁷': ' to the seventh power ',
        '⁸': ' to the eighth power ',
        '⁹': ' to the ninth power ',
        '¹': ' to the first power ',
        '⁰': ' to the zeroth power ',
        
        # Fractions
        '½': ' one half ',
        '⅓': ' one third ',
        '⅔': ' two thirds ',
        '¼': ' one quarter ',
        '¾': ' three quarters ',
        '⅕': ' one fifth ',
        '⅖': ' two fifths ',
        '⅗': ' three fifths ',
        '⅘': ' four fifths ',
        '⅙': ' one sixth ',
        '⅚': ' five sixths ',
        '⅛': ' one eighth ',
        '⅜': ' three eighths ',
        '⅝': ' five eighths ',
        '⅞': ' seven eighths ',
        
        # Comparison operators
        '<': ' less than ',
        '>': ' greater than ',
        '≤': ' less than or equal to ',
        '≥': ' greater than or equal to ',
        '≠': ' not equal to ',
        '≈': ' approximately equal to ',
        '≡': ' identical to ',
        
        # Currency symbols
        '$': ' dollar ',
        '€': ' euro ',
        '£': ' pound ',
        '¥': ' yen ',
        '¢': ' cent ',
        
        # Other common symbols
        '°': ' degrees ',
        '©': ' copyright ',
        '®': ' registered ',
        '™': ' trademark ',
        '§': ' section ',
        '¶': ' paragraph ',
        '†': ' dagger ',
        '‡': ' double dagger ',
        '•': ' bullet ',
        '→': ' arrow ',
        '←': ' left arrow ',
        '↑': ' up arrow ',
        '↓': ' down arrow ',
        '↔': ' double arrow ',
        '⟨': ' left angle bracket ',
        '⟩': ' right angle bracket ',
        '…': ' ellipsis ',
        '‰': ' per mille ',
        '‱': ' per ten thousand ',
    }
    
    result = text
    for symbol, replacement in replacements.items():
        result = result.replace(symbol, replacement)
    
    return result

def add_commas_to_abbreviations(text):
    """
    Find abbreviations (consecutive uppercase letters) and add commas between them
    """
    # Pattern to match sequences of 2 or more consecutive uppercase letters
    # This will match things like AM, NYPD, FBI, etc.
    abbreviation_pattern = r'\b([A-Z]{2,})\b'
    
    def add_commas(match):
        abbrev = match.group(1)
        # Add commas between each letter: "NYPD" -> "N, Y, P, D"
        return ', '.join(abbrev)
    
    return re.sub(abbreviation_pattern, add_commas, text)

def process_function_file(input_file, output_file=None):
    """
    Process a Python file containing function definitions
    
    Args:
        input_file (str): Path to the input Python file
        output_file (str): Path to save the cleaned file (optional)
    
    Returns:
        str: The cleaned content
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned_content = clean_function_text(content)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"Cleaned file saved to: {output_file}")
        
        return cleaned_content
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

# Example usage and testing
if __name__ == "__main__":
    # Test the cleaning function with sample text
    sample_function = """
def example_function(input_data):
    # This function processes NYC data 🏙️ with 50% accuracy
    result = process_data(input_data) ⚡️some_function({"action": "process", "data": "sample"})
    
    # Calculate area: length² + width³ = total area
    area = length**2 + width**3  # Mathematical operations
    price = $150.75 + €25.50  # Currency calculations
    
    # Handle AM/PM times and NYPD data 👮‍♂️
    if temperature > 32°F or humidity ≥ 75%:
        result = handle_weather_data(result) ⚡️another_call({"time": True})
    
    # Process FBI and CIA data 🕵️ @ headquarters
    agencies = ["FBI", "CIA", "NSA"]
    url = "https://example.com/data"  # API endpoint
    email = "admin@agency.gov" & "backup@fbi.gov"
    percentage = 75% ± 5%
    
    # Mathematical formulas with symbols
    formula = "√(x² + y²) ≈ distance"
    ratio = ¾ of total_amount
    pi_value = π * radius²
    
    # Comparison operations
    if value1 < value2 and result ≠ expected:
        difference = value2 - value1
    
    api_result = fetch_data() ⚡️random_name({"endpoint": "/api/data"})
    return result 🎉 = = = = = = = 


    hola Hola  i love the NYPD and i love cheese π(E=mc²) is the best formula
"""

    print("Original function:")
    print(sample_function)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_function_text(sample_function)
    print("Cleaned function:")
    print(cleaned)
    
    print("\n" + "="*50 + "\n")
    
    # Example of processing a file
    # Uncomment the lines below to process an actual file
    # cleaned_content = process_function_file('input_function.py', 'cleaned_function.py')
    
    # You can also process without saving to a file
    # cleaned_content = process_function_file('input_function.py')
    # print(cleaned_content)