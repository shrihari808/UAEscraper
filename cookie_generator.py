import time
import json
import undetected_chromedriver as uc

def generate_cookies():
    """
    Launches a browser, allows for manual login, and saves session cookies.
    """
    print("🚀 Starting cookie generation process...")
    print("   A browser window will open. Please log into LinkedIn manually.")
    
    driver = None
    try:
        driver = uc.Chrome(use_subprocess=True)
        driver.get("https://www.linkedin.com/login")
        
        print("\n⏳ Please log in now. The script is waiting for you to land on the feed page...")
        
        # Wait until the user has logged in and the feed page is loaded
        while "feed" not in driver.current_url:
            time.sleep(1)
            
        # Add a small delay to ensure all cookies are set
        time.sleep(5)
        
        # Get cookies from the current session
        cookies = driver.get_cookies()
        
        # Save cookies to a file
        with open("cookies.json", "w") as f:
            json.dump(cookies, f)
            
        print("\n✅ Cookies have been saved to cookies.json!")
        print("   You can now close the browser window and run the main scraper script.")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
    finally:
        if driver:
            # The script will hang here until you manually close the browser
            # This is intentional to give you time to see the confirmation message.
            input("\nPress Enter to exit...")
            driver.quit()

if __name__ == "__main__":
    generate_cookies()
