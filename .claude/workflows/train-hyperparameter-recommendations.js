export const meta = {
  name: 'train-hyperparameter-recommendations',
  description: 'Add dataset-aware training hyperparameter recommendations with reasons, shown as advisory hints in the UI',
  phases: [
    { title: 'Explore', detail: 'map training UI, dataset counting, and adjust logic' },
    { title: 'Implement', detail: 'add recommendation function and UI hints' },
    { title: 'Verify', detail: 'syntax + consistency + UX review' },
  ],
}

const REPO = '/Users/koil/Google-Teachable-Machine-TFLite-model-training/AItraining'

const EXPLORE_SCHEMA = {
  type: 'object',
  properties: {
    uiStructure: { type: 'string', description: 'how _render_train_config builds each input (exact code with line numbers, verbatim)' },
    datasetCounters: { type: 'string', description: 'exact functions available to count samples per class / total, with signatures and line numbers' },
    adjustLogic: { type: 'string', description: 'current _adjust_cfg_for_dataset_size logic in trainer.py, verbatim with line numbers' },
    callers: { type: 'string', description: 'who calls _render_train_config and _adjust_cfg_for_dataset_size, with line numbers' },
  },
  required: ['uiStructure', 'datasetCounters', 'adjustLogic', 'callers'],
}

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    changesMade: { type: 'array', items: { type: 'string' } },
    syntaxStatus: { type: 'string' },
  },
  required: ['changesMade', 'syntaxStatus'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    syntaxOk: { type: 'boolean' },
    issues: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['syntaxOk', 'issues', 'summary'],
}

const SPEC = `
GOAL: In the training settings UI (Streamlit _render_train_config in app.py), show RECOMMENDED hyperparameters computed from the actual dataset, with a one-line REASON for each recommendation. The recommendation is ADVISORY ONLY: the user sees the hint but can keep typing their own values. Also add a single "Use recommended settings" button that applies all recommendations at once.

RECOMMENDATION RULES (compute from dataset counts):
- total = number of sample files across all classes (images only: .png/.jpg/.jpeg/.bmp)
- train_count = round(total * (1 - validation_split)), min 1
- recommended batch_size: min(user_batch, max(2, train_count // 4)) — ensures >=4 steps per epoch
- steps_per_epoch = ceil(train_count / batch)
- recommended epochs: clamp(ceil(800 / steps_per_epoch), 30, 300) — target ~800 total updates
- recommended learning_rate: 0.0008 if total < 100 else 0.001
- recommended validation_split: 0.15 if total < 60 else 0.2
- img_size / conv filters / dense units / optimizer: no recommendation (leave user's), but MAY note img_size should match preprocessing (96).

REASONS must be human-readable, e.g.:
- batch: "only 68 training samples → 48 gives 1 step/epoch; recommend ≥4 steps/epoch"
- epochs: "1 step/epoch × 30 epochs = 30 updates; recommend ~800 updates"
- lr: "small dataset (<100) — lower lr reduces oscillation"
- val_split: "only 55 samples — keep more for training"

IMPLEMENTATION REQUIREMENTS:
1. trainer.py: add a PUBLIC function recommend_train_params(total_samples: int, cfg: TrainConfig) -> tuple[dict, list[str]] that returns (recommended_kwargs dict with keys matching TrainConfig fields, reasons list in the same order). Reuse the numeric logic from _adjust_cfg_for_dataset_size where possible — refactor if it avoids duplication. DO NOT change _adjust_cfg_for_dataset_size behavior.
2. app.py: in _render_train_config(cfg), before building inputs:
   - count total samples via existing dataset helpers (find them; e.g. _tm_dataset_dir() and file listing — reuse whatever record_controller/dataset_io helpers exist, or _list_class_image_files equivalent)
   - compute recommendations via recommend_train_params
   - under each relevant input (batch_size, epochs, learning_rate, validation_split) add st.caption(f"💡 Recommended: {value} — {reason}") ONLY when the current cfg value differs from the recommendation (or always show if simple)
   - add a button "Use recommended settings" that sets st.session_state.train_cfg fields to the recommended values and calls st.rerun()
   - add a small st.info at top: "Recommendations are advisory — you can keep your own values."
3. record_controller.py: minimal or no change (only touch if the train button should log the recommendation; skip unless trivially safe).

CONSTRAINTS:
- Match existing code style (indentation, st.* idioms, type hints, comment density).
- Do NOT break training flow: _render_train_config must still return a TrainConfig; the button must not loop infinitely (use a sentinel in session_state or rely on st.rerun with updated cfg values).
- All Chinese/English user-facing text: use English like the rest of the UI.
`

const explorePrompt = (f) => `
Read ${REPO}/${f} fully, plus the other two files, and report:
${SPEC}

Report ONLY what this file contains that is relevant: uiStructure / datasetCounters / adjustLogic / callers per the schema. Verbatim code with line numbers for everything that will be edited.

Return raw structured data only.
`

const implPrompt = (f, explore) => `
Explore findings: ${JSON.stringify(explore, null, 2)}

${SPEC}

EDIT TASK: Edit ${REPO}/${f} directly (Read + Edit tools) to implement the parts of the spec that belong to this file. Then run a syntax check (py_compile for trainer.py/record_controller.py, ast.parse for app.py). Report changes made and syntax status.

Return raw structured data only.
`

const verifyPrompt = (f, impl) => `
The implement agent reported: ${JSON.stringify(impl, null, 2)}

VERIFY: 1) re-run the syntax check on ${REPO}/${f}; 2) read all three files and check cross-file consistency: recommend_train_params signature matches its callers, the button/rereun flow in app.py has no infinite-loop risk, reasons list length matches recommended dict keys, and no existing behavior changed. Report issues + summary.

Return raw structured data only.
`

phase('Explore')
const explores = await parallel(['trainer.py', 'app.py', 'record_controller.py'].map((f) => () =>
  agent(explorePrompt(f), { label: `explore:${f}`, phase: 'Explore', schema: EXPLORE_SCHEMA })
))

phase('Implement')
const impls = await parallel(['trainer.py', 'app.py', 'record_controller.py'].map((f, i) => () =>
  agent(implPrompt(f, explores[i] || {}), { label: `impl:${f}`, phase: 'Implement', schema: IMPL_SCHEMA })
))

phase('Verify')
const vers = await parallel(['trainer.py', 'app.py', 'record_controller.py'].map((f, i) => () =>
  agent(verifyPrompt(f, impls[i] || {}), { label: `verify:${f}`, phase: 'Verify', schema: VERIFY_SCHEMA })
))

log('Workflow complete')
return { explores, impls, vers }