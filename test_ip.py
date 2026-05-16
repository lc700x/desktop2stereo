import requests

def is_ip_in_china():
    try:
        # Get your public IP
        ip = requests.get("https://api.ipify.org").text.strip()
        
        # Query geolocation info from ip-api.com
        response = requests.get(f"http://ip-api.com/json/{ip}", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        country = data.get("country", "")
        
        print(f"Your IP: {ip}, Country: {country}")
        return country == "China"
    except Exception as e:
        print(f"Error checking IP location: {e}")
        return False

if is_ip_in_china():
    print("Your IP is in China")
else:
    print("Your IP is NOT in China")
