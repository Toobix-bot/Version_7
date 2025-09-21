import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

function readApiKeyFromEnvFile(): string | null {
  try {
    const envPath = path.join(process.cwd(), '.env');
    if (!fs.existsSync(envPath)) return null;
    const content = fs.readFileSync(envPath, 'utf-8');
    const line = content.split(/\r?\n/).find(l => l.trim().startsWith('API_TOKENS='));
    if (!line) return null;
    const val = line.split('=')[1]?.trim() ?? '';
    const clean = val.replace(/^"|^'|"$|'$/g, '');
    const first = clean.split(',')[0]?.trim();
    return first || null;
  } catch {
    return null;
  }
}

const API_KEY = process.env.E2E_API_KEY || readApiKeyFromEnvFile() || 'test';

// Minimal smoke: open UI, set key, verify dashboard visible and badges update
test('UI smoke: key + dashboard', async ({ page }) => {
  const base = process.env.E2E_BASE_URL; // optional override; otherwise use baseURL from config

  // Prevent blocking prompt and seed API key before any scripts run
  await page.addInitScript(([k]) => {
    try { (window as any).prompt = () => k; } catch {}
    try { localStorage.setItem('api_key', k); } catch {}
  }, [API_KEY]);

  await page.goto((base || '') + '/story/ui');
  // Force dashboard pane selection to avoid hidden sections
  const dashBtn = page.locator('#navBtnDashboard');
  if (await dashBtn.count()) { await dashBtn.click(); }

  // Wait for dashboard badges to be present
  await page.locator('#dashSse').waitFor({ state: 'attached', timeout: 5000 });
  await page.locator('#dashMetrics').waitFor({ state: 'attached', timeout: 5000 });
  // Wait until badges change from '?' (up to 12s)
  await page.waitForFunction(() => {
    const sse = document.getElementById('dashSse');
    const met = document.getElementById('dashMetrics');
    return !!sse && !!met && sse.textContent && met.textContent &&
      !sse.textContent.includes('?') && !met.textContent.includes('?');
  }, undefined, { timeout: 12_000 });
});
