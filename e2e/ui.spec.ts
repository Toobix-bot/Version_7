import { test, expect } from '@playwright/test';

const API_KEY = process.env.E2E_API_KEY || 'test';

// Minimal smoke: open UI, set key, verify dashboard visible and badges update
test('UI smoke: key + dashboard', async ({ page }) => {
  const base = process.env.E2E_BASE_URL || 'http://127.0.0.1:8003';

  // Prevent blocking prompt and seed API key before any scripts run
  await page.addInitScript(([k]) => {
    try { (window as any).prompt = () => k; } catch {}
    try { localStorage.setItem('api_key', k); } catch {}
  }, [API_KEY]);

  await page.goto(base + '/story/ui');
  // Force dashboard pane selection to avoid hidden sections
  const dashBtn = page.locator('#navBtnDashboard');
  if (await dashBtn.count()) { await dashBtn.click(); }

  // Give the app a moment to connect SSE/metrics and update badges
  await page.waitForTimeout(1500);
  // Wait for dashboard badges to be present
  await page.locator('#dashSse').waitFor({ state: 'attached', timeout: 5000 });
  await page.locator('#dashMetrics').waitFor({ state: 'attached', timeout: 5000 });
  const sseText = await page.locator('#dashSse').innerText();
  const metText = await page.locator('#dashMetrics').innerText();

  expect(sseText).not.toContain('?');
  expect(metText).not.toContain('?');
});
