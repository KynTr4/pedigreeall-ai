import sys, shutil, os, subprocess, time, json, base64
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Setup DB
sys.path.insert(0, 'tests')
from test_race_day_dashboard import RaceDayDashboardTests
x = RaceDayDashboardTests()
x.setUp()
p = Path('scratch/global_browser.db')
p.parent.mkdir(exist_ok=True)
p.unlink(missing_ok=True)
shutil.copy2(x.db, p)
x.tearDown()

# Environment for web app
os.environ['DB_PATH'] = str(p.absolute())
os.environ['LOG_DIR'] = str(Path('scratch').absolute())
os.environ['WEB_HOST'] = '127.0.0.1'
os.environ['WEB_PORT'] = '18087'
os.environ['WEB_USERNAME'] = 'admin'
os.environ['WEB_PASSWORD'] = 'testpass'

# Start web app
print("Starting web app...")
proc = subprocess.Popen([sys.executable, 'web_app.py'], cwd=os.getcwd(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)

# Run selenium
o = Options()
o.binary_location = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
o.add_argument('--headless=new')
o.add_argument('--no-sandbox')
o.add_argument('--window-size=1500,1000')
o.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

try:
    print("Starting Chrome...")
    d = webdriver.Chrome(options=o)
    d.execute_cdp_cmd('Network.enable', {})
    d.execute_cdp_cmd('Network.setExtraHTTPHeaders', {'headers': {'Authorization': 'Basic ' + base64.b64encode(b'admin:testpass').decode()}})
    out = {}
    
    for path in ('races', 'performance', 'diagnostics', 'bet-simulator', 'logs'):
        url = f'http://127.0.0.1:18087/{path}'
        print(f"Visiting {url}")
        d.get(url)
        time.sleep(5)
        
        if path in ('races', 'performance'):
            field = 'race-day-date' if path == 'races' else 'filter-date'
            try:
                el = d.find_element('id', field)
                d.execute_script("arguments[0].value='2026-06-28';arguments[0].dispatchEvent(new Event('change'))", el)
                time.sleep(4)
            except Exception as e:
                print(f"Error setting date on {path}: {e}")
        
        out[path] = {'title': d.title, 'body_rows': len(d.find_elements('css selector', 'tbody tr'))}
    
    logs = [x for x in d.get_log('browser') if x['level'] == 'SEVERE' and 'favicon.ico' not in x['message']]
    out['logs'] = logs
    print(json.dumps(out, ensure_ascii=False, indent=2))
    
    if logs:
        print("FAIL: SEVERE logs found!")
        sys.exit(1)
    else:
        print("PASS: No SEVERE logs found.")
finally:
    try:
        d.quit()
    except:
        pass
    proc.kill()
    p.unlink(missing_ok=True)
