"""
Playwright browser automation wrapper.

Responsibilities:
  1. setup_session()     — login to sandbox + select Queen's Consolidated merchant
                           returns a storage_state dict (auth cookies/localStorage)
  2. run_test_case()     — execute one TestCase's steps in an isolated browser context
  3. run_all_test_cases() — run all test cases with up to 3 concurrent browser sessions

Why storage_state instead of re-logging in per test:
  Login takes 3-4 seconds per test case. With 10 test cases that's 30-40 wasted
  seconds. We log in once, save the browser session to a dict, and every test case
  starts from a fresh isolated context that is already authenticated.

Why isolated contexts per test case:
  Each test case gets its own browser context — like a fresh incognito window loaded
  with the auth cookies. Steps in TC-001 cannot affect the DOM seen by TC-002.
  This prevents test ordering from mattering.

Why asyncio.Semaphore(3):
  We spawn all test cases concurrently but limit to 3 running at any moment.
  This matches the architecture design and avoids overwhelming the sandbox.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

from loguru import logger
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from qa_agent.config import get_settings
from qa_agent.models import TestCase, TestResult, TestStep, TestStepType

# ── Constants ─────────────────────────────────────────────────────────────────
STEP_TIMEOUT_MS   = 10_000   # 10s per individual step
NAV_TIMEOUT_MS    = 30_000   # 30s for page navigations
TEST_TIMEOUT_S    = 90       # 90s hard cap per test case
CONCURRENCY       = 3        # max simultaneous browser contexts
SCREENSHOTS_DIR   = Path(__file__).parent.parent.parent / "screenshots"


# ── Session setup ─────────────────────────────────────────────────────────────

async def setup_session(browser: Browser, run_id: str) -> dict:
    """
    Log in to the Friendbuy sandbox and select Queen's Consolidated merchant.

    Returns a storage_state dict — Playwright's serialised representation of
    auth cookies and localStorage. Pass this to browser.new_context() so
    test cases start already authenticated.

    Raises RuntimeError if login or merchant selection fails — the run
    should abort rather than execute tests in an unauthenticated state.
    """
    settings = get_settings()
    context = await browser.new_context(
        viewport={"width": 1280, "height": 800},
        ignore_https_errors=True,   # sandbox may have a self-signed cert
    )
    page = await context.new_page()
    page.set_default_timeout(NAV_TIMEOUT_MS)

    try:
        login_url = settings.sandbox_url.rstrip("/") + "/login"
        logger.info(f"[{run_id}] setup_session: navigating to {login_url}")
        await page.goto(login_url, wait_until="domcontentloaded")

        # ── Fill login form ───────────────────────────────────────────────────
        # Use resilient locators — try common patterns for email/password fields
        await _fill_login_form(page, settings.sandbox_username, settings.sandbox_password)

        # ── Submit ────────────────────────────────────────────────────────────
        await _submit_login(page)

        # ── Wait for post-login navigation ────────────────────────────────────
        await page.wait_for_load_state("domcontentloaded")
        logger.info(f"[{run_id}] setup_session: logged in, selecting merchant")

        # ── Select Queen's Consolidated merchant ──────────────────────────────
        await _select_merchant(page, settings.sandbox_merchant_name, run_id)

        # ── Capture authenticated state ───────────────────────────────────────
        storage_state = await context.storage_state()
        logger.info(f"[{run_id}] setup_session: session ready")
        return storage_state

    except Exception as exc:
        # Capture a screenshot of whatever state we're in before failing
        _ensure_screenshots_dir(run_id)
        fail_path = SCREENSHOTS_DIR / run_id / "session_setup_failure.png"
        try:
            await page.screenshot(path=str(fail_path))
            logger.error(f"[{run_id}] Session setup failure screenshot: {fail_path}")
        except Exception:
            pass
        raise RuntimeError(f"Sandbox session setup failed: {exc}") from exc
    finally:
        await context.close()


async def _fill_login_form(page: Page, username: str, password: str) -> None:
    """
    Fill the login form using resilient locators.
    Tries multiple selector strategies so minor UI changes don't break auth.
    """
    # Email field — try in order of specificity
    email_locator = page.locator(
        "input[type=email], input[name=email], input[name=username], input[placeholder*=email i]"
    ).first
    await email_locator.wait_for(state="visible", timeout=STEP_TIMEOUT_MS)
    await email_locator.fill(username)

    # Password field
    password_locator = page.locator(
        "input[type=password], input[name=password]"
    ).first
    await password_locator.fill(password)


async def _submit_login(page: Page) -> None:
    """Click the login submit button using resilient locators."""
    submit_locator = page.locator(
        "button[type=submit], "
        "input[type=submit], "
        "button:has-text('Sign in'), "
        "button:has-text('Log in'), "
        "button:has-text('Login')"
    ).first
    await submit_locator.click()


async def _select_merchant(page: Page, merchant_name: str, run_id: str) -> None:
    """
    Select the correct merchant after login.

    Friendbuy's sandbox shows a merchant selection step after auth.
    We look for the merchant name as text and click it.
    If a different UI pattern is used (dropdown, search box), the locator
    strategies below handle the most common cases.
    """
    try:
        # Most common pattern: merchant name appears as a clickable item
        merchant_locator = page.get_by_text(merchant_name, exact=False).first
        await merchant_locator.wait_for(state="visible", timeout=STEP_TIMEOUT_MS)
        await merchant_locator.click()
        await page.wait_for_load_state("domcontentloaded")
        logger.debug(f"[{run_id}] Merchant '{merchant_name}' selected")
    except PlaywrightTimeoutError:
        # Merchant selection page may not appear if already defaulted.
        # Log a warning but continue — some sandbox configs skip this step.
        logger.warning(
            f"[{run_id}] Merchant selection not found — may already be set. Continuing."
        )


# ── Test case execution ───────────────────────────────────────────────────────

async def run_test_case(
    tc: TestCase,
    browser: Browser,
    storage_state: dict,
    run_id: str,
) -> TestResult:
    """
    Execute one TestCase in an isolated browser context.

    Each call creates a fresh context loaded with the auth storage_state,
    executes every step in order, captures a screenshot on failure, and
    returns a TestResult regardless of pass/fail.
    """
    _ensure_screenshots_dir(run_id)
    start = time.monotonic()
    context: Optional[BrowserContext] = None

    try:
        context = await browser.new_context(
            storage_state=storage_state,
            viewport={"width": 1280, "height": 800},
            ignore_https_errors=True,
        )
        page = await context.new_page()
        page.set_default_timeout(STEP_TIMEOUT_MS)
        page.set_default_navigation_timeout(NAV_TIMEOUT_MS)

        logger.info(f"[{run_id}] Running {tc.id}: {tc.title}")

        last_screenshot: Optional[str] = None

        for step_index, step in enumerate(tc.steps, start=1):
            try:
                screenshot_path = await _execute_step(
                    page, step, tc.id, step_index, run_id
                )
                if screenshot_path:
                    last_screenshot = screenshot_path
            except Exception as exc:
                duration = time.monotonic() - start
                error_msg = f"Step {step_index} ({step.type.value}): {exc}"
                logger.warning(f"[{run_id}] {tc.id} FAILED — {error_msg}")

                # Always screenshot on failure — this is what the QA report shows
                fail_path = str(SCREENSHOTS_DIR / run_id / f"{tc.id}_fail.png")
                try:
                    await page.screenshot(path=fail_path, full_page=True)
                    last_screenshot = fail_path
                except Exception:
                    pass

                return TestResult(
                    test_case_id=tc.id,
                    passed=False,
                    error_message=error_msg,
                    screenshot_path=last_screenshot,
                    duration_seconds=round(duration, 2),
                )

        duration = time.monotonic() - start
        logger.info(f"[{run_id}] {tc.id} PASSED in {duration:.1f}s")
        return TestResult(
            test_case_id=tc.id,
            passed=True,
            screenshot_path=last_screenshot,
            duration_seconds=round(duration, 2),
        )

    except asyncio.TimeoutError:
        duration = time.monotonic() - start
        return TestResult(
            test_case_id=tc.id,
            passed=False,
            error_message=f"Test case timed out after {TEST_TIMEOUT_S}s",
            duration_seconds=round(duration, 2),
        )
    finally:
        if context:
            await context.close()


async def _execute_step(
    page: Page,
    step: TestStep,
    tc_id: str,
    step_index: int,
    run_id: str,
) -> Optional[str]:
    """
    Execute one TestStep and return a screenshot path if one was taken.
    Raises on assertion failures or Playwright errors.
    """
    logger.debug(
        f"[{run_id}] {tc_id} step {step_index}: "
        f"{step.type.value} — {step.description}"
    )

    if step.type == TestStepType.NAVIGATE:
        await page.goto(step.value, wait_until="domcontentloaded")

    elif step.type == TestStepType.CLICK:
        await page.locator(step.selector).first.click()

    elif step.type == TestStepType.FILL:
        await page.locator(step.selector).first.fill(step.value or "")

    elif step.type == TestStepType.ASSERT_TEXT:
        # Check that expected text is present anywhere in the matched element
        locator = page.locator(step.selector).first
        await locator.wait_for(state="visible", timeout=STEP_TIMEOUT_MS)
        actual = await locator.inner_text()
        if step.expected and step.expected not in actual:
            raise AssertionError(
                f"Expected text '{step.expected}' not found in element '{step.selector}'. "
                f"Got: '{actual[:200]}'"
            )

    elif step.type == TestStepType.ASSERT_VISIBLE:
        locator = page.locator(step.selector).first
        await locator.wait_for(state="visible", timeout=STEP_TIMEOUT_MS)

    elif step.type == TestStepType.WAIT:
        ms = int(step.value or "1000")
        ms = min(ms, 3000)    # enforce the 3s max from the prompt rules
        await asyncio.sleep(ms / 1000)

    elif step.type == TestStepType.SCREENSHOT:
        label = step.description.replace(" ", "_").lower()[:40]
        path = str(SCREENSHOTS_DIR / run_id / f"{tc_id}_step{step_index}_{label}.png")
        await page.screenshot(path=path, full_page=True)
        logger.debug(f"[{run_id}] Screenshot saved: {path}")
        return path

    return None


# ── Concurrent runner ─────────────────────────────────────────────────────────

async def run_all_test_cases(
    test_cases: list[TestCase],
    run_id: str,
    staging_url: str,
) -> list[TestResult]:
    """
    Run all test cases with up to CONCURRENCY=3 simultaneous browser sessions.

    Flow:
      1. Launch one shared browser
      2. Run setup_session() once → get storage_state dict
      3. Run every test case concurrently, gated by a Semaphore(3)
      4. Return results in the same order as input test_cases

    Why one shared browser:
      Browser startup is expensive (~500ms). Sharing the process across all
      test cases saves time. Each test case still gets its own isolated context.
    """
    _ensure_screenshots_dir(run_id)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )

        try:
            # Single login for the whole run
            storage_state = await setup_session(browser, run_id)
        except RuntimeError as exc:
            logger.error(f"[{run_id}] Aborting: {exc}")
            await browser.close()
            # Return all test cases as errors so the pipeline can continue
            return [
                TestResult(
                    test_case_id=tc.id,
                    passed=False,
                    error_message=str(exc),
                    duration_seconds=0.0,
                )
                for tc in test_cases
            ]

        async def _run_with_semaphore(tc: TestCase) -> TestResult:
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        run_test_case(tc, browser, storage_state, run_id),
                        timeout=TEST_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[{run_id}] {tc.id} hard timeout after {TEST_TIMEOUT_S}s")
                    return TestResult(
                        test_case_id=tc.id,
                        passed=False,
                        error_message=f"Hard timeout after {TEST_TIMEOUT_S}s",
                        duration_seconds=float(TEST_TIMEOUT_S),
                    )

        results = await asyncio.gather(
            *[_run_with_semaphore(tc) for tc in test_cases]
        )

        await browser.close()

    passed = sum(1 for r in results if r.passed)
    logger.info(
        f"[{run_id}] All tests complete — "
        f"{passed}/{len(results)} passed"
    )
    return list(results)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_screenshots_dir(run_id: str) -> None:
    """Create screenshots/{run_id}/ if it doesn't exist."""
    (SCREENSHOTS_DIR / run_id).mkdir(parents=True, exist_ok=True)
