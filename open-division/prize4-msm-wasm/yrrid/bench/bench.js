var seleniumWebdriver = require('selenium-webdriver');
var chrome = require('selenium-webdriver/chrome');
var logging = require('selenium-webdriver/lib/logging');
const capabilities = require('selenium-webdriver/lib/capabilities');
const TIMEOUT = 30000000;
var options = new chrome.Options().headless();
options.addArguments("start-maximized"); // https://stackoverflow.com/a/26283818/1689770
options.addArguments("enable-automation"); // https://stackoverflow.com/a/43840128/1689770
options.addArguments("--headless"); // only if you are ACTUALLY running headless
options.addArguments("--no-sandbox"); //https://stackoverflow.com/a/50725918/1689770
options.addArguments("--disable-dev-shm-usage"); //https://stackoverflow.com/a/50725918/1689770
options.addArguments("--disable-browser-side-navigation"); //https://stackoverflow.com/a/49123152/1689770
options.addArguments("--disable-gpu"); //https://stackoverflow.com/questions/51959986/how-to-solve-selenium-chromedriver-timed-out-receiving-message-from-renderer-exc
options.addArguments("--disable-infobars");
options.addArguments("--disable-translate");
options.addArguments("--disable-extensions");
var prefs = new logging.Preferences();
prefs.setLevel(logging.Type.BROWSER, logging.Level.INFO);
var Capabilities = capabilities.Capabilities;
var caps = Capabilities.chrome();
caps.setLoggingPrefs(prefs);
var By = seleniumWebdriver.By;

var driver = new seleniumWebdriver.Builder()
  .forBrowser('chrome')
  .setChromeOptions(options)
  .build();

driver.manage()
  .setTimeouts({
    implicit: TIMEOUT, pageLoad:
      TIMEOUT, script: TIMEOUT
  })
  .then(() => driver.manage().getTimeouts())
  .then((val) => {
    console.log(val);
    const element = By.id('wasm-msm');
    driver
      .get(`http://localhost:8080`)
      .then(() => driver.findElement(element))
      .then(elem => elem.getText())
      .then(val => console.log(val))
      .then(() => driver.quit());
  });
