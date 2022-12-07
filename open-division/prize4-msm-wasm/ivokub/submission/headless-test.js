const puppeteer = require('puppeteer');

(async () => {
  // there are several ways to run the test -- with the built-in chromium or
  // external Chrome. Locally built-in chromium works okay, but it fails in
  // CoreWeave. In CoreWeave, manually start chrome with
  // `google-chrome --headless --disable-dev-shm-usage --remote-debugging-port=7777`
  const browser = await puppeteer.launch();
  //const browser = await puppeteer.connect({'browserURL': 'http://localhost:7777'});
  const page = await browser.newPage();
  await page.goto('http://localhost:8080/', {'timeout': 0});
  await page.waitForFunction(
    'document.getElementById("wasm-msm").textContent.includes("Correctness check passed")',
    {'timeout': 0},
  );
  const content = await page.$eval('#wasm-msm', el => el.textContent);
  console.log(content);
  // when using external browser, then instead of closing disconnect from it.
  // PS! It may happen that the test stays running in Chrome tab. Then I
  // recommend manually closing Chrome and restarting it. You can see this
  // happening when Chrome is 100% CPU usage in htop.
  await browser.close();
  //await browser.disconnect();
})();