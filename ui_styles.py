def inject_teachable_style() -> None:
    import streamlit as st

    st.markdown(
        """
<style>
  :root {
    --tm-bg: var(--background-color, #f5f7fa);
    --tm-flow-bg: color-mix(in srgb, var(--tm-bg) 88%, #f5f7fa 12%);
    --tm-surface: var(--secondary-background-color, #ffffff);
    --tm-surface-strong: var(--secondary-background-color, #ffffff);
    --tm-text: var(--text-color, #23262d);
    --tm-muted: color-mix(in srgb, var(--text-color, #23262d) 58%, transparent);
    --tm-border: color-mix(in srgb, var(--text-color, #23262d) 14%, transparent);
    --tm-primary: var(--primary-color, #2f73ea);
    --tm-primary-strong: color-mix(in srgb, var(--primary-color, #2f73ea) 88%, black 12%);
    --tm-accent: color-mix(in srgb, var(--primary-color, #2f73ea) 10%, var(--secondary-background-color, #ffffff));
    --tm-success: #0f9f6e;
    --tm-shadow: 0 2px 8px rgba(17, 24, 39, 0.10);
    --tm-shadow-soft: 0 2px 8px rgba(17, 24, 39, 0.08);
    --tm-card-radius: 8px;
  }

  .stApp {
    background: var(--tm-bg);
    color: var(--tm-text);
  }

  [data-testid="stAppViewContainer"] > .main {
    background: transparent;
  }

  [data-testid="stAppViewContainer"] > .main,
  [data-testid="stAppViewContainer"] > .main > div,
  .block-container {
    width: 100%;
    max-width: none !important;
    box-sizing: border-box;
  }
  .block-container {
    padding-top: 0.65rem;
    padding-bottom: 2rem;
    padding-left: 18px;
    padding-right: 18px;
  }
  .tm-top-brand-marker {
    display: none;
  }
  div:has(.tm-top-brand-marker) {
    position: fixed;
    top: 12px;
    left: 18px;
    z-index: 2000;
    margin: 0;
  }
  div:has(.tm-top-brand-marker) .stButton > button {
    min-height: 48px !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    background: var(--tm-surface) !important;
    border: 0 !important;
    box-shadow: var(--tm-shadow) !important;
    color: var(--tm-primary-strong) !important;
    font-weight: 800 !important;
    font-size: 18px !important;
    outline: none !important;
  }
  div:has(.tm-top-brand-marker) .stButton > button:focus,
  div:has(.tm-top-brand-marker) .stButton > button:focus-visible {
    outline: none !important;
  }
  div:has(.tm-top-brand-marker) .stButton > button:hover {
    background: color-mix(in srgb, var(--tm-surface) 86%, var(--tm-accent)) !important;
  }
  .tm-top-brand {
    display: inline-flex;
    align-items: center;
    gap: 14px;
    min-height: 48px;
    margin: 0 0 18px 0;
    padding: 10px 18px;
    border-radius: 0 0 14px 14px;
    background: var(--tm-surface);
    border: 1px solid var(--tm-border);
    box-shadow: var(--tm-shadow-soft);
  }
  .tm-top-brand-menu {
    font-size: 20px;
    color: var(--tm-muted);
    line-height: 1;
  }
  .tm-top-brand-title {
    font-size: 18px;
    font-weight: 800;
    color: var(--tm-primary-strong);
    letter-spacing: -0.02em;
  }

  [data-testid="stHeader"] {
    background: transparent;
  }
  [data-testid="stToolbar"],
  [data-testid="stToolbarActions"],
  [data-testid="stHeaderActionElements"],
  #MainMenu {
    display: none !important;
  }

  h1, h2, h3, h4, h5, h6, p, label {
    color: var(--tm-text);
  }

  .tm-title { font-size: 44px; font-weight: 800; margin: 0 0 10px 0; letter-spacing: -0.03em; }
  .tm-sub { color: var(--tm-muted); margin: 0 0 20px 0; font-size: 16px; line-height: 1.6; }
  .tm-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--tm-primary);
    margin-bottom: 8px;
  }
  .tm-hero {
    border: 1px solid var(--tm-border);
    border-radius: 20px;
    padding: 20px 24px;
    background: var(--tm-surface);
    box-shadow: var(--tm-shadow);
    margin-bottom: 16px;
  }
  .tm-hero-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.9fr);
    gap: 18px;
    align-items: start;
  }
  .tm-hero-compact {
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(520px, 1fr);
    gap: 14px;
    align-items: center;
  }
  .tm-hero-stats {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
  }
  .tm-flow-step {
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 56px;
    padding: 10px 12px;
    border-radius: 14px;
    background: rgba(255,255,255,0.96);
    border: 1px solid var(--tm-border);
    box-shadow: var(--tm-shadow-soft);
  }
  .tm-flow-step span {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 999px;
    background: rgba(49, 105, 255, 0.10);
    color: var(--tm-primary-strong);
    font-size: 12px;
    font-weight: 800;
    flex: 0 0 auto;
  }
  .tm-flow-step strong {
    display: block;
    font-size: 14px;
    line-height: 1.2;
  }
  .tm-flow-step small {
    display: block;
    margin-top: 2px;
    color: var(--tm-muted);
    font-size: 11px;
    line-height: 1.35;
  }
  .tm-node-head {
    display: flex;
    align-items: center;
    min-height: 36px;
    margin-bottom: 8px;
  }
  .tm-node-head-classes {
    margin-bottom: 10px;
  }
  .tm-node-head-train {
    margin-bottom: 10px;
    justify-content: flex-start;
  }
  .tm-node-head-preview {
    margin-bottom: 8px;
  }
  .tm-node-head h3 {
    margin: 0;
    font-size: 16px;
    line-height: 1.2;
    letter-spacing: -0.02em;
  }
  .tm-hero-copy h2 {
    margin: 0 0 10px 0;
    font-size: 24px;
    line-height: 1.2;
    letter-spacing: -0.03em;
  }
  .tm-hero-copy p {
    margin: 0;
    color: var(--tm-muted);
    font-size: 14px;
    line-height: 1.45;
  }
  .tm-hero-panel {
    border-radius: 16px;
    padding: 14px 16px;
    background: var(--tm-surface);
    border: 1px solid var(--tm-border);
    box-shadow: var(--tm-shadow-soft);
  }
  .tm-hero-panel h4 {
    margin: 0 0 14px 0;
    font-size: 15px;
    font-weight: 700;
  }
  .tm-stat-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
  }
  .tm-stat {
    border-radius: 14px;
    padding: 12px 14px;
    background: var(--tm-surface);
    border: 1px solid var(--tm-border);
  }
  .tm-stat-label {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--tm-muted);
    margin-bottom: 6px;
  }
  .tm-stat-value {
    font-size: 20px;
    font-weight: 800;
    line-height: 1.1;
  }
  .tm-stat-note {
    margin-top: 6px;
    font-size: 13px;
    color: var(--tm-muted);
  }
  .tm-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 16px;
  }
  .tm-chip {
    display: inline-flex;
    align-items: center;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(49, 105, 255, 0.08);
    color: var(--tm-primary-strong);
    font-size: 12px;
    font-weight: 600;
  }
  .tm-card {
    border: 1px solid var(--tm-border);
    border-radius: 18px;
    padding: 18px 18px 16px 18px;
    background: var(--tm-surface);
    box-shadow: var(--tm-shadow);
    height: 100%;
  }
  .tm-card h3 { margin: 0 0 8px 0; font-size: 22px; }
  .tm-card p { margin: 0 0 12px 0; color: var(--tm-muted); line-height: 1.6; }
  .tm-badge {
    display: inline-block;
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(16, 35, 63, 0.06);
    color: var(--tm-muted);
    margin-left: 8px;
  }
  .tm-card-footnote {
    color: var(--tm-muted);
    font-size: 13px;
  }
  .tm-section-head {
    margin-bottom: 10px;
  }
  .tm-section-head h3 {
    margin: 0 0 6px 0;
    font-size: 20px;
    letter-spacing: -0.02em;
  }
  .tm-section-head p {
    margin: 0;
    color: var(--tm-muted);
    line-height: 1.6;
  }
  .tm-class-header {
    padding: 10px 12px;
    margin: 12px 0 8px 0;
    border-radius: 14px;
    background: var(--tm-surface);
    border: 1px solid var(--tm-border);
    box-shadow: var(--tm-shadow-soft);
  }
  .tm-class-header strong {
    font-size: 17px;
  }
  .tm-class-meta {
    font-size: 13px;
    color: var(--tm-muted);
    margin-top: -2px;
  }
  .tm-inline-note {
    border-radius: 14px;
    padding: 12px 14px;
    background: rgba(255,255,255,0.94);
    border: 1px solid var(--tm-border);
    color: var(--tm-muted);
    margin: 8px 0 12px 0;
  }
  .tm-class-stack {
    position: relative;
  }
  .tm-class-card-marker {
    display: none;
  }
  div[data-testid="stVerticalBlock"]:has(.tm-class-card-marker) {
    position: relative;
    margin-bottom: 28px;
    padding: 0;
    border-radius: var(--tm-card-radius);
    border: 0;
    background: var(--tm-surface);
    box-shadow: var(--tm-shadow);
    overflow: hidden;
  }
  div[data-testid="stVerticalBlock"]:has(.tm-class-card-marker)::after,
  div[data-testid="stVerticalBlock"]:has(.tm-class-card-marker)::before {
    display: none;
  }
  div[data-testid="stVerticalBlock"]:has(.tm-class-card-marker) [data-testid="column"] {
    padding-left: 10px;
    padding-right: 10px;
  }
  div[data-testid="stVerticalBlock"]:has(.tm-class-card-marker) > [data-testid="stHorizontalBlock"]:first-of-type {
    padding: 12px 0 10px 0;
    background: var(--tm-surface);
  }
  div[data-testid="stVerticalBlock"]:has(.tm-class-card-marker) > [data-testid="stHorizontalBlock"]:first-of-type .stButton > button {
    min-height: 34px !important;
    border-radius: 6px !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    color: var(--tm-muted) !important;
    font-size: 16px !important;
    padding: 0 !important;
    outline: none !important;
  }
  div[data-testid="stVerticalBlock"]:has(.tm-class-card-marker) > [data-testid="stHorizontalBlock"]:first-of-type .stButton > button:focus,
  div[data-testid="stVerticalBlock"]:has(.tm-class-card-marker) > [data-testid="stHorizontalBlock"]:first-of-type .stButton > button:focus-visible {
    outline: none !important;
  }
  div[data-testid="stVerticalBlock"]:has(.tm-class-card-marker) > [data-testid="stHorizontalBlock"]:first-of-type .stButton > button:hover {
    background: rgba(0, 0, 0, 0.04) !important;
    color: var(--tm-text) !important;
  }
  div[data-testid="stVerticalBlock"]:has(.tm-class-card-marker) > [data-testid="stHorizontalBlock"]:nth-of-type(2) [data-testid="stButton"] {
    display: flex;
    justify-content: center;
  }
  div[data-testid="stVerticalBlock"]:has(.tm-class-card-marker) > [data-testid="stHorizontalBlock"]:nth-of-type(2) .stButton > button {
    width: 76px !important;
    height: 76px !important;
    min-height: 76px !important;
    border-radius: 4px !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    white-space: pre-line !important;
    line-height: 1.15 !important;
    padding: 8px 6px !important;
    text-align: center !important;
    border: 0 !important;
    background: rgba(26, 115, 232, 0.10) !important;
    color: var(--tm-primary-strong) !important;
    box-shadow: none !important;
    outline: none !important;
  }
  div[data-testid="stVerticalBlock"]:has(.tm-class-card-marker) > [data-testid="stHorizontalBlock"]:nth-of-type(2) .stButton > button:hover {
    background: rgba(26, 115, 232, 0.16) !important;
  }
  .tm-class-card {
    position: relative;
    margin-bottom: 24px;
    padding: 0;
    border-radius: 14px;
    border: 1px solid var(--tm-border);
    background: var(--tm-surface);
    box-shadow: var(--tm-shadow);
    overflow: hidden;
  }
  .tm-class-card::after {
    content: "";
    position: absolute;
    top: 50%;
    right: -24px;
    width: 24px;
    height: 2px;
    background: color-mix(in srgb, var(--tm-text) 12%, transparent);
    transform: translateY(-1px);
  }
  .tm-class-card::before {
    content: "";
    position: absolute;
    top: calc(50% - 16px);
    right: -28px;
    width: 28px;
    height: 32px;
    border-top: 2px solid color-mix(in srgb, var(--tm-text) 12%, transparent);
    border-right: 2px solid color-mix(in srgb, var(--tm-text) 12%, transparent);
    border-bottom: 2px solid color-mix(in srgb, var(--tm-text) 12%, transparent);
    border-left: 0;
    border-top-right-radius: 16px;
    border-bottom-right-radius: 16px;
  }
  .tm-class-card [data-testid="column"] {
    padding-left: 10px;
    padding-right: 10px;
  }
  .tm-class-card [data-testid="stHorizontalBlock"] {
    align-items: center;
  }
  .tm-class-card > [data-testid="stHorizontalBlock"]:first-of-type {
    padding: 12px 0 10px 0;
    background: var(--tm-surface);
  }
  .tm-class-card > [data-testid="stHorizontalBlock"]:nth-of-type(2) {
    padding: 0 0 14px 0;
  }
  .tm-class-edit {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 34px;
    color: var(--tm-muted);
    font-size: 18px;
  }
  .tm-class-title-field [data-testid="stTextInputRootElement"] > div,
  .tm-class-title-field [data-baseweb="base-input"] > div {
    background: transparent !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    min-height: 36px;
  }
  .tm-class-title-field input {
    padding-left: 0 !important;
    padding-right: 0 !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
    color: var(--tm-text) !important;
    background: transparent !important;
    outline: none !important;
    box-shadow: none !important;
  }
  .tm-class-title-field input:focus,
  .tm-class-title-field input:focus-visible {
    outline: none !important;
    box-shadow: none !important;
  }
  .tm-class-title-field [data-testid="stTextInputRootElement"] > div:focus-within,
  .tm-class-title-field [data-baseweb="base-input"] > div:focus-within {
    background: color-mix(in srgb, var(--tm-primary) 8%, transparent) !important;
    border-color: color-mix(in srgb, var(--tm-primary) 16%, transparent) !important;
    box-shadow: none !important;
    padding-left: 10px !important;
    padding-right: 10px !important;
  }
  .tm-class-title-field input::placeholder {
    color: var(--tm-text) !important;
    opacity: 0.92;
  }
  .tm-class-title-text {
    display: flex;
    align-items: center;
    min-height: 34px;
  }
  .tm-class-title-text h3 {
    margin: 0;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -0.01em;
  }
  .tm-camera-note {
    margin-top: 10px;
    font-size: 12px;
    color: var(--tm-muted);
    line-height: 1.35;
  }
  .tm-class-count {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 34px;
    padding: 0 12px;
    border-radius: 999px;
    background: color-mix(in srgb, var(--tm-primary) 5%, var(--tm-surface));
    color: var(--tm-muted);
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
  }
  .tm-class-divider {
    height: 1px;
    margin: 0;
    background: var(--tm-border);
  }
  .tm-class-subhead {
    margin: 2px 0 10px 0;
    font-size: 15px;
    font-weight: 600;
    color: var(--tm-text);
    text-transform: none;
    letter-spacing: 0;
  }
  .tm-add-class-wrap .stButton > button {
    border-style: dashed;
    border-width: 1px;
    min-height: 52px;
    background: transparent;
    color: var(--tm-muted) !important;
  }
  .tm-add-class-marker {
    display: none;
  }
  div[data-testid="stVerticalBlock"]:has(.tm-add-class-marker) {
    background: transparent;
    border: 0;
    border-radius: 0;
    padding: 0;
  }
  div[data-testid="stVerticalBlock"]:has(.tm-add-class-marker) .stButton > button {
    min-height: 56px !important;
    border-radius: 8px !important;
    border: 2px dashed color-mix(in srgb, var(--tm-text) 18%, transparent) !important;
    background: transparent !important;
    color: var(--tm-muted) !important;
    font-weight: 700 !important;
    outline: none !important;
  }
  div[data-testid="stVerticalBlock"]:has(.tm-add-class-marker) .stButton > button:hover {
    background: rgba(0, 0, 0, 0.03) !important;
  }
  .tm-mini-stat {
    margin-bottom: 8px;
    padding: 10px 12px;
    border-radius: 12px;
    background: color-mix(in srgb, var(--tm-surface) 94%, var(--tm-accent));
    border: 1px solid var(--tm-border);
    font-size: 14px;
    font-weight: 700;
    color: var(--tm-text);
  }
  .tm-preview-note {
    min-height: 54px;
    border-radius: 12px;
    border: 1px solid var(--tm-border);
    background: color-mix(in srgb, var(--tm-surface) 94%, var(--tm-accent));
    padding: 10px 12px;
    color: var(--tm-muted);
    line-height: 1.35;
    font-size: 13px;
  }
  .tm-class-card .stButton > button[kind="secondary"] {
    border-radius: 12px;
  }
  .tm-class-card .stButton > button {
    min-height: 58px;
    border-radius: 6px;
    font-size: 12px !important;
    font-weight: 700 !important;
  }
  .tm-class-card .stButton > button:not([kind="primary"]) {
    background: color-mix(in srgb, var(--tm-primary) 12%, var(--tm-surface)) !important;
    color: var(--tm-primary-strong) !important;
    border: 1px solid color-mix(in srgb, var(--tm-primary) 18%, transparent) !important;
  }
  .tm-class-card [data-testid="stButton"] > button {
    white-space: pre-line !important;
    line-height: 1.15 !important;
    padding-top: 8px !important;
    padding-bottom: 8px !important;
  }
  .tm-class-upload [data-testid="stPopoverButton"] {
    width: 100%;
  }
  .tm-class-upload [data-testid="stPopoverButton"] {
    display: flex;
    justify-content: center;
  }
  .tm-class-upload [data-testid="stPopoverButton"] button {
    width: 76px !important;
    height: 76px !important;
    min-height: 76px !important;
    border-radius: 4px !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    background: rgba(26, 115, 232, 0.10) !important;
    color: var(--tm-primary-strong) !important;
    border: 0 !important;
    white-space: pre-line !important;
    line-height: 1.15 !important;
    padding: 8px 6px !important;
    outline: none !important;
  }
  .tm-class-upload [data-testid="stPopoverButton"] button p {
    color: inherit !important;
  }
  .tm-class-upload [data-testid="stPopoverButton"] svg {
    display: none !important;
  }
  .tm-class-menu [data-testid="stPopoverButton"] button {
    min-height: 34px;
    padding-left: 0 !important;
    padding-right: 0 !important;
    border-radius: 6px !important;
    font-size: 16px !important;
    line-height: 1 !important;
    background: transparent !important;
    color: var(--tm-muted) !important;
    border: 0 !important;
    box-shadow: none !important;
  }
  .tm-class-menu [data-testid="stPopoverButton"] button:hover {
    background: rgba(0, 0, 0, 0.04) !important;
    color: var(--tm-text) !important;
  }
  .tm-class-menu [data-testid="stPopoverButton"] button p {
    color: inherit !important;
  }
  .tm-steps {
    margin: 12px 0 18px 0;
    color: var(--tm-muted);
    font-size: 14px;
  }
  .tm-step-on {
    font-weight: 800;
    color: var(--tm-primary-strong);
  }
  .tm-kv {
    border-radius: 16px;
    padding: 14px 16px;
    background: rgba(49, 105, 255, 0.06);
    border: 1px solid rgba(49, 105, 255, 0.10);
  }
  .tm-panel {
    border: 1px solid var(--tm-border);
    border-radius: 16px;
    padding: 10px;
    background: var(--tm-surface-strong);
    box-shadow: var(--tm-shadow-soft);
  }
  .tm-capture-panel {
    margin-top: 12px;
    padding: 10px;
    border-radius: 14px;
    background: color-mix(in srgb, var(--tm-surface) 94%, var(--tm-accent));
  }
  .tm-capture-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 8px;
  }
  .tm-capture-head strong {
    font-size: 14px;
    font-weight: 800;
  }
  .tm-capture-stage {
    padding: 8px;
    border-radius: 12px;
    background: color-mix(in srgb, var(--tm-primary) 9%, var(--tm-surface));
    min-height: 100%;
  }
  .tm-capture-side-head {
    margin-bottom: 8px;
    font-size: 13px;
    font-weight: 700;
    color: var(--tm-muted);
  }
  [data-testid="column"] > div:has(.tm-node-head-classes) {
    padding: 0;
    background: transparent;
    border: 0;
    box-shadow: none;
    overflow: visible;
  }
  .tm-train-card-marker,
  .tm-preview-card-marker {
    display: none;
  }
  [data-testid="column"] > div:has(.tm-train-card-marker),
  [data-testid="column"] > div:has(.tm-preview-card-marker) {
    padding: 16px 18px;
    background: var(--tm-surface);
    border: 0;
    border-radius: var(--tm-card-radius);
    box-shadow: var(--tm-shadow);
    overflow: visible;
    margin-top: 0;
  }
  [data-testid="column"] > div:has(.tm-train-card-marker)::before,
  [data-testid="column"] > div:has(.tm-train-card-marker)::after,
  [data-testid="column"] > div:has(.tm-preview-card-marker)::before {
    display: none;
  }

  .tm-card-head h3 {
    margin: 0 0 10px 0;
    font-size: 14px;
    font-weight: 700;
  }
  .tm-card-note {
    margin-top: 10px;
    color: var(--tm-muted);
    font-size: 12px;
    line-height: 1.35;
  }
  .tm-footer {
    position: fixed;
    right: 18px;
    bottom: 12px;
    font-size: 12px;
    color: var(--tm-muted);
    z-index: 1000;
    pointer-events: none;
  }

  .tm-layout-wrap-marker,
  .tm-layout-row-marker {
    display: none;
  }
  div[data-testid="stVerticalBlock"]:has(.tm-layout-wrap-marker) {
    position: relative;
    background: var(--tm-flow-bg);
    padding: 80px 40px 92px 40px;
    border-radius: 12px;
    overflow: visible;
  }
  div[data-testid="stVerticalBlock"]:has(.tm-layout-wrap-marker) .tm-flow-svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    z-index: 0;
    pointer-events: none;
  }
  div[data-testid="stVerticalBlock"]:has(.tm-layout-wrap-marker) .tm-flow-svg path {
    stroke: #c5c8cc;
    stroke-width: 2;
    fill: none;
  }
  div[data-testid="stHorizontalBlock"]:has(.tm-layout-row-marker) {
    position: relative;
    z-index: 1;
    align-items: center;
    justify-content: center;
    gap: 64px !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.tm-layout-row-marker) > div[data-testid="column"]:nth-child(1) {
    flex: 0 0 520px !important;
    max-width: 520px !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.tm-layout-row-marker) > div[data-testid="column"]:nth-child(2) {
    flex: 0 0 240px !important;
    max-width: 240px !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.tm-layout-row-marker) > div[data-testid="column"]:nth-child(3) {
    flex: 0 0 340px !important;
    max-width: 340px !important;
  }

  .tm-classified-browse-marker {
    display: none;
  }
  div[data-testid="stHorizontalBlock"]:has(.tm-classified-browse-marker) {
    align-items: end;
  }
  div[data-testid="stHorizontalBlock"]:has(.tm-classified-browse-marker) > div[data-testid="column"]:nth-child(2) {
    display: flex;
    align-items: end;
  }
  div[data-testid="stHorizontalBlock"]:has(.tm-classified-browse-marker) > div[data-testid="column"]:nth-child(2) > div {
    width: 100%;
  }
  div[data-testid="stHorizontalBlock"]:has(.tm-classified-browse-marker) > div[data-testid="column"]:nth-child(2) .stButton {
    margin-top: 0;
  }
  div[data-testid="stHorizontalBlock"]:has(.tm-classified-browse-marker) > div[data-testid="column"]:nth-child(2) .stButton > button {
    min-height: 40px;
  }

  div[data-testid="column"]:has(.tm-preview-card-marker) .stButton > button {
    background: color-mix(in srgb, var(--tm-text) 6%, transparent) !important;
    border: 0 !important;
    border-radius: 4px !important;
    color: var(--tm-muted) !important;
    font-weight: 700 !important;
  }
  div[data-testid="column"]:has(.tm-preview-card-marker) .stButton > button:hover {
    background: color-mix(in srgb, var(--tm-text) 9%, transparent) !important;
    color: var(--tm-text) !important;
  }
  div[data-testid="column"]:has(.tm-train-card-marker) [data-testid="stPopoverButton"] button {
    background: transparent !important;
    border: 0 !important;
    color: var(--tm-muted) !important;
    box-shadow: none !important;
    justify-content: flex-start !important;
  }
  div[data-testid="column"]:has(.tm-train-card-marker) [data-testid="stPopoverButton"] button:hover {
    background: color-mix(in srgb, var(--tm-text) 6%, transparent) !important;
    color: var(--tm-text) !important;
  }
  [data-testid="stPopover"] {
    width: 100%;
  }
  .tm-status {
    display: inline-flex;
    align-items: center;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
  }
  .tm-status-ok {
    background: rgba(15, 159, 110, 0.12);
    color: var(--tm-success);
  }
  .tm-status-warn {
    background: rgba(255, 168, 0, 0.14);
    color: #9d6500;
  }
  .tm-status-bad {
    background: rgba(226, 62, 87, 0.14);
    color: #b4233d;
  }
  .tm-status-idle {
    background: rgba(16, 35, 63, 0.08);
    color: var(--tm-muted);
  }

  [data-testid="stHorizontalBlock"] > [data-testid="column"] > div {
    gap: 0.9rem;
  }

  [data-testid="stTextInputRootElement"] > div,
  [data-testid="stNumberInputContainer"] > div,
  [data-testid="stSelectbox"] > div,
  [data-testid="stTextArea"] textarea,
  [data-testid="stFileUploader"] section,
  [data-baseweb="select"] > div,
  [data-baseweb="base-input"] > div {
    border-radius: 16px !important;
  }

  [data-testid="stTextInputRootElement"] > div,
  [data-testid="stNumberInputContainer"] > div,
  [data-baseweb="base-input"] > div,
  [data-baseweb="select"] > div,
  [data-testid="stFileUploader"] section {
    background: var(--tm-surface) !important;
    border: 1px solid var(--tm-border) !important;
    box-shadow: none !important;
  }

  [data-testid="stTextArea"] textarea {
    background: var(--tm-surface) !important;
    border: 1px solid var(--tm-border) !important;
    box-shadow: none !important;
  }

  .stButton > button {
    border-radius: 10px;
    border: 1px solid var(--tm-border);
    background: var(--tm-surface);
    color: var(--tm-text) !important;
    font-weight: 700;
    min-height: 40px;
    white-space: nowrap;
    box-shadow: none;
  }

  .stButton > button p {
    white-space: nowrap;
    color: inherit !important;
  }

  .stButton > button[kind="primary"] {
    border: 0 !important;
    background: var(--tm-primary) !important;
    color: #ffffff !important;
  }

  .stButton > button[kind="primary"] p {
    color: #ffffff !important;
  }

  .stButton > button:hover {
    border-color: rgba(47, 115, 234, 0.32);
    color: var(--tm-primary-strong);
  }

  .stButton > button[kind="primary"]:hover {
    color: #ffffff !important;
    background: var(--tm-primary-strong) !important;
  }

  [data-testid="stMetric"] {
    background: var(--tm-surface);
    border: 1px solid var(--tm-border);
    border-radius: 14px;
    padding: 12px 16px;
    box-shadow: var(--tm-shadow-soft);
  }

  [data-testid="stExpander"] {
    border: 1px solid var(--tm-border) !important;
    border-radius: 12px !important;
    background: var(--tm-surface) !important;
    overflow: hidden;
  }
  [data-testid="stExpander"] summary {
    padding-top: 0.2rem !important;
    padding-bottom: 0.2rem !important;
    min-height: 2.4rem !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    color: var(--tm-text) !important;
  }
  [data-testid="stExpander"] summary:hover {
    background: color-mix(in srgb, var(--tm-primary) 7%, transparent) !important;
  }
  [data-testid="stExpander"] details > div {
    padding: 0.55rem 0.65rem 0.65rem 0.65rem !important;
    background: var(--tm-surface) !important;
    border-top: 1px solid var(--tm-border) !important;
  }
  [data-testid="stExpander"] label {
    color: var(--tm-text) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
  }
  [data-testid="stExpander"] input,
  [data-testid="stExpander"] textarea {
    color: var(--tm-text) !important;
  }
  [data-testid="stExpander"] [data-testid="stTextInputRootElement"] > div,
  [data-testid="stExpander"] [data-testid="stNumberInputContainer"] > div,
  [data-testid="stExpander"] [data-testid="stSelectbox"] > div,
  [data-testid="stExpander"] [data-baseweb="select"] > div,
  [data-testid="stExpander"] [data-baseweb="base-input"] > div {
    background: color-mix(in srgb, var(--tm-surface) 88%, var(--tm-bg)) !important;
    border: 1px solid var(--tm-border) !important;
  }
  [data-testid="stExpander"] [data-baseweb="select"] div,
  [data-testid="stExpander"] [data-baseweb="select"] span,
  [data-testid="stExpander"] [data-baseweb="select"] p {
    color: var(--tm-text) !important;
  }

  [data-testid="stAlert"] {
    border-radius: 16px;
  }

  .stTabs [data-baseweb="tab-list"] {
    gap: 10px;
  }

  .stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    background: rgba(255,255,255,0.78);
    padding: 10px 16px;
  }

  .stTabs [aria-selected="true"] {
    background: rgba(49, 105, 255, 0.12);
    color: var(--tm-primary-strong);
  }

  @media (max-width: 1080px) {
    .tm-hero-compact,
    .tm-hero-grid {
      grid-template-columns: 1fr;
    }
    .tm-hero-stats {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .tm-class-card::after {
      display: none;
    }
    .tm-class-card::before {
      display: none;
    }
    div[data-testid="stVerticalBlock"]:has(.tm-layout-wrap-marker) {
      padding: 18px 16px 54px 16px;
    }
    div[data-testid="stVerticalBlock"]:has(.tm-layout-wrap-marker) .tm-flow-svg {
      display: none;
    }
    div[data-testid="stHorizontalBlock"]:has(.tm-layout-row-marker) {
      flex-direction: column;
      align-items: stretch;
      gap: 18px !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.tm-layout-row-marker) > div[data-testid="column"]:nth-child(1),
    div[data-testid="stHorizontalBlock"]:has(.tm-layout-row-marker) > div[data-testid="column"]:nth-child(2),
    div[data-testid="stHorizontalBlock"]:has(.tm-layout-row-marker) > div[data-testid="column"]:nth-child(3) {
      flex: 1 1 auto !important;
      max-width: none !important;
    }
  }

</style>
        """,
        unsafe_allow_html=True,
    )


def inject_dark_mode_patch() -> None:
    """Dark-mode patch for the Streamlit chrome.

    The embedded workspace (app.py's HTML blob) already follows the system
    theme via prefers-color-scheme, but Streamlit itself has no dark theme
    configured — in macOS dark mode the page stayed light behind dark
    components. Align the Streamlit surfaces with the blob's dark palette
    and let component iframes show the page background through.
    """
    import streamlit as st

    st.markdown(
        """
<style>
  /* The embedded workspace iframe must never paint its own background. */
  .stComponent, .stComponent iframe,
  [data-testid="stComponentBody"],
  [data-testid="element-container"] .stComponent {
    background: transparent !important;
  }

  @media (prefers-color-scheme: dark) {
    /* Palette mirrors the workspace blob's own dark variables. */
    .stApp, [data-testid="stAppViewContainer"] {
      background: #0f141a;
      color: #e8eaed;
    }
    [data-testid="stAppViewContainer"] > .main,
    .block-container,
    section[data-testid="stSidebar"] {
      background: transparent;
      color: #e8eaed;
    }
    :root {
      --tm-bg: #0f141a;
      --tm-flow-bg: #111821;
      --tm-surface: #1a212b;
      --tm-surface-strong: #202833;
      --tm-text: #e8eaed;
      --tm-muted: #a8b0ba;
      --tm-border: rgba(255, 255, 255, 0.12);
      --tm-accent: rgba(138, 180, 248, 0.10);
      --tm-shadow: 0 2px 8px rgba(0, 0, 0, 0.32);
      --tm-shadow-soft: 0 2px 8px rgba(0, 0, 0, 0.24);
    }
    .stApp p, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
      color: var(--tm-text);
    }
  }
</style>
""",
        unsafe_allow_html=True,
    )


def inject_workspace_fullcanvas() -> None:
    """Pin the embedded workspace to a full-viewport, non-scrolling canvas.

    The workspace blob renders inside a Streamlit component iframe sized to
    its content; together with the page scrollbar that produced a fixed-width
    layout with endless blank space below. Restore the original Teachable
    Machine behavior: the iframe fills the window and the page never scrolls.
    Only injected on the workspace page.
    """
    import streamlit as st

    st.markdown(
        """
<style>
  [data-testid="stAppViewContainer"] > .main,
  section[data-testid="stSidebar"] {
    overflow: hidden !important;
  }
  [data-testid="stAppViewContainer"] > .main > div,
  .block-container {
    height: 100vh;
    max-height: 100vh;
    overflow: hidden !important;
    padding: 0 !important;
    margin: 0 !important;
    max-width: none !important;
  }
  .stComponent iframe,
  iframe.stIFrame,
  [data-testid="stComponentBody"] iframe,
  iframe[title^="streamlit_component"] {
    height: 100vh !important;
    min-height: 100vh !important;
    max-height: 100vh !important;
    display: block;
  }
</style>
""",
        unsafe_allow_html=True,
    )
