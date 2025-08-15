"""
privacy_validator.py

This module implements a Privacy & Security Validator using non–ML/DL approaches.
It validates an LLM-generated response by checking for sensitive data (PII)
and security-related terms using regular expression patterns loaded from a CSV dataset.
If the dataset cannot be downloaded, an embedded default dataset is used.

Dependencies:
  - requests (install via: pip install requests)
  - Built-in modules: re, csv, io

Usage:
  Simply import the validate_privacy(response: str) function.
  It returns True if the response is free from disallowed sensitive content,
  and False otherwise.
"""

import re
import csv
import io
import requests

# URL to a CSV dataset containing PII regex patterns.
# The CSV should have two columns: "type" and "pattern"
# (Replace with a valid URL if available)
DATASET_URL = "https://raw.githubusercontent.com/example/pii_dataset/main/sample_pii.csv"

# Embedded CSV data (fallback) as a multi-line string.
# Expanded to include additional patterns for credit card numbers, IP addresses, and US passport numbers.
EMBEDDED_CSV = """type,pattern
SSN,\\b\\d{3}-\\d{2}-\\d{4}\\b
Email,\\b[\\w\\.-]+@[\\w\\.-]+\\.\\w+\\b
Phone,\\b\\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b
CreditCard,\\b(?:\\d[ -]*?){13,16}\\b
IPAddress,\\b(?:(?:2[0-4]\\d|25[0-5]|1\\d\\d|[1-9]?\\d)\\.){3}(?:2[0-4]\\d|25[0-5]|1\\d\\d|[1-9]?\\d)\\b
Passport,\\b[A-Z][0-9]{8}\\b
DriverLicense,\\b[A-Z]{1}\\d{7}\\b
TaxID,\\b\\d{2}-\\d{7}\\b
MedicalRecordNumber,\\bMRN\\d{6}\\b
StudentID,\\bS\\d{7}\\b
EmployeeID,\\bE\\d{6}\\b
BankAccount,\\b\\d{10,12}\\b
RoutingNumber,\\b\\d{9}\\b
VIN,\\b[A-HJ-NPR-Z0-9]{17}\\b
IMEI,\\b\\d{15}\\b
MACAddress,\\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\\b
NationalID,\\b\\d{9}\\b
BirthCertificateNumber,\\bBC\\d{8}\\b
VoterID,\\bV\\d{7}\\b
UtilityBillNumber,\\bUB\\d{6}\\b
InsurancePolicyNumber,\\bIP\\d{8}\\b
CreditReportNumber,\\bCRN\\d{6}\\b
PhotoID,\\bPID\\d{6}\\b
SocialInsuranceNumber,\\b\\d{3}-\\d{3}-\\d{3}\\b
HealthInsuranceNumber,\\bHIN\\d{8}\\b
ProfessionalLicenseNumber,\\bPLN\\d{6}\\b
DriverPermitNumber,\\bDP\\d{7}\\b
MilitaryID,\\bMID\\d{8}\\b
GovernmentID,\\bGID\\d{6}\\b
VoterRegistrationNumber,\\bVRN\\d{7}\\b
VisaNumber,\\bVISA\\d{8}\\b
CardExpirationDate,\\b(0[1-9]|1[0-2])/[0-9]{2}\\b
IBAN,\\b[A-Z]{2}\\d{2}[A-Z0-9]{1,30}\\b
BIC,\\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\\b
AccountNumber,\\b\\d{8,12}\\b
BankCardNumber,\\b(?:\\d[ -]*?){13,16}\\b
MobileNumber,\\b\\+?\\d{1,3}[-.\\s]?\\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b
AlternativeEmail,\\b[\\w\\.-]+@[\\w\\.-]+\\.\\w+\\b
ContactNumber,\\b\\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b
FaxNumber,\\b\\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b
TollFreeNumber,\\b1[-.\\s]?800[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b
SocialSecurityAlternative,\\b\\d{3}[-\\s]?\\d{2}[-\\s]?\\d{4}\\b
PassportAlternative,\\b[A-Z]{2}\\d{7}\\b
EmployeeNumber,\\bEMP\\d{5}\\b
StudentNumber,\\bSTU\\d{6}\\b
CreditScore,\\bCS\\d{3}\\b
LoanAccountNumber,\\bLA\\d{8}\\b
MembershipID,\\bMID\\d{7}\\b
InsuranceClaimNumber,\\bIC\\d{7}\\b
CustomerID,\\bCUST\\d{6}\\b
"""


# List of security-related terms that are disallowed.
SECURITY_TERMS = {"hack", "exploit", "malware", "phish", "ransomware"}

def load_csv_from_string(csv_string: str) -> list:
    """
    Loads and compiles regex patterns from an embedded CSV string.
    
    Args:
      csv_string (str): The CSV content as a string.
    
    Returns:
      List of compiled regex patterns.
    """
    patterns = []
    reader = csv.DictReader(io.StringIO(csv_string))
    for row in reader:
        pattern_str = row.get("pattern", "").strip()
        if pattern_str:
            try:
                compiled_pattern = re.compile(pattern_str)
                patterns.append(compiled_pattern)
            except re.error as e:
                print(f"Regex compilation error for pattern '{pattern_str}': {e}")
    return patterns

def load_pii_patterns(url: str) -> list:
    """
    Attempts to load a CSV file from the provided URL and compile regex patterns.
    The CSV is expected to have headers "type" and "pattern".
    Returns a list of compiled regex patterns.
    If download or parsing fails, returns patterns from the embedded CSV.
    
    Args:
      url (str): The URL to load the CSV dataset.
    
    Returns:
      List of compiled regex patterns.
    """
    patterns = []
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raises HTTPError for bad status codes.
        csv_content = response.text
        reader = csv.DictReader(io.StringIO(csv_content))
        for row in reader:
            pattern_str = row.get("pattern", "").strip()
            if pattern_str:
                try:
                    compiled_pattern = re.compile(pattern_str)
                    patterns.append(compiled_pattern)
                except re.error as e:
                    print(f"Regex compilation error for pattern '{pattern_str}': {e}")
        if patterns:
            return patterns
    except Exception as e:
        print(f"Failed to load dataset from URL '{url}': {e}")
    
    # Fallback: load patterns from the embedded CSV string.
    print("Loading PII patterns from embedded dataset...")
    return load_csv_from_string(EMBEDDED_CSV)

# Load the PII patterns once at module load time.
PII_PATTERNS = load_pii_patterns(DATASET_URL)

def validate_privacy(response: str) -> bool:
    """
    Validates the response text for privacy and security issues.
    Returns True if no disallowed sensitive data or security terms are found;
    otherwise returns False.
    
    Args:
      response (str): The LLM-generated response text.
    """
    lower_response = response.lower()

    # 1. Check against all PII patterns.
    for pattern in PII_PATTERNS:
        if pattern.search(response):
            print(f"PII match found: Pattern '{pattern.pattern}'")
            return False

    # 2. Check for disallowed security terms.
    for term in SECURITY_TERMS:
        if term in lower_response:
            print(f"Security-sensitive term detected: '{term}'")
            return False

    return True

# Example usage (for testing purposes only; remove these lines when integrating):
# response_example = "The technician's email is john.doe@example.com, phone is 123-456-7890, and his passport number is A12345678."
# print(validate_privacy(response_example))
