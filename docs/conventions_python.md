
# Python Engineering Conventions
**Version:** 2.4.0
**Audience:** Human engineers **and** AI assistants (machine-readable examples included)
**Last Updated:** 2026-01-21

### Document and Project Versioning
Projects MUST follow Semantic Versioning (SemVer):
- Always update document and project versioning on change
- Projects MUST follow SemVer starting at v1.0.0.
- During pre-MVP (0.y.z), breaking changes MUST bump minor (0.Y.z) and backward-compatible fixes bump patch (0.y.Z).
SemVer:
- **Major (X.y.z):** Breaking changes
- **Minor (x.Y.z):** Backward-compatible feature additions
- **Patch (x.y.Z):** Backward-compatible bug fixes or refactors

### Convention Scope

This document defines **generic, reusable conventions** for Python development.


**Important:**
These conventions are org-wide standards (apply to all projects), but remain project-agnostic (do not reference any one project’s requirements).

## 1. Core Principles

- **Proactive Communication**: If an implementation requires a deviation from established patterns or conventions, you MUST ask for permission first. Detail the necessary change and the reason for it.

- **Development Plan**: Before adding complex features or refactoring multiple files
  (defined as touching more than 2 files or >200 lines of change),
  briefly outline a plan.

- **Fail Fast (Pre-MVP):** Code MUST surface invalid states early and loudly.
   - Silent fallbacks, placeholder defaults, and error-swallowing patterns MUST NOT be used.
   - If defaults are necessary, they MUST be explicitly documented and MUST NOT mask errors.

## 2. Python Code Conventions

### 2.1. General Syntax and Style
- **Guiding Principles**: All code MUST be clear, Pythonic, and explicit. Adhere strictly to PEP 8 (Style Guide) and PEP 20 (The Zen of Python). Readability and simplicity are more important than premature optimization or overly terse code.
- **Python Version**: Code MUST be compatible with Python 3.12 or newer. Use modern language features and avoid deprecated forms.
- **Imports**: Imports SHOULD be ordered and grouped according to PEP 8 standards: Standard Library, then third-party packages, then local application/library specific imports. Use a tool like `isort` to enforce this automatically.
  - Prefer absolute imports for all non-local modules.
  - Relative imports MAY be used within a tightly scoped package.
- `from x import *` MUST NOT be used (except in __init__.py with explicit intent).

### 2.2. File and Module Structure
To ensure consistency and readability across the project, Python files (.py) SHOULD follow this standard layout from top to bottom, Not all sections are required in every file; omit sections that are not applicable:
- **Shebang** (for executable scripts only, e.g., #!/usr/bin/env python3)
- **Module Docstring**: A string literal explaining the module's purpose.
- **Imports**: Grouped and ordered as specified in section 2.1.
- **Module "Dunder" Variables**: __all__, __author__, __version__, etc.
- **Constants**: Module-level constants in UPPER_SNAKE_CASE.
- **Type Definitions**: Pydantic models, TypeDicts, or type aliases used in the module.
- **Classes**: All class definitions for the module.
- **Module-Level Functions**: Functions that are not part of any class.
- **Main Block**: The if __name__ == "__main__": block for executable scripts.

### 2.3. Design and Structure
- **Simplicity (KISS)**: Prefer simple, direct solutions. Avoid unnecessary complexity or over-engineering.
- **Minimal Features (YAGNI)**: "You Ain't Gonna Need It." Only implement functionality when it is actually required, rather than when you just foresee that it might be needed in the future.
- **Don't Repeat Yourself (DRY)**: Refactor repeated code blocks or logic into reusable functions, methods, or constants.
- **Single Responsibility Principle (SRP)**: Design functions, methods, and classes to have a single, clear, and focused purpose. Decompose complex functions into smaller, well-defined helper functions.
- **Separation of Concerns (SoC)**: Organize classes and files to separate functionalities strictly. For example:
   - processing logic MUST reside in a dedicated processing module.
   - UI code MUST NOT contain processing logic.
   - Data access and manipulation SHOULD be handled by a dedicated data handler/service layer.
- **The Principle of Least Astonishment (POLA)**: Design components, interfaces, and code in a way that behaves exactly as a user or another developer would expect it to behave, minimizing surprises.

### 2.4. Naming Conventions
- **Descriptiveness**: Use descriptive and unambiguous names for all variables, functions, classes, and modules.
- **Casing**:
  - `PascalCase` for classes.
  - `snake_case` for functions, methods, variables, and modules.
  - `UPPER_SNAKE_CASE` for constants.
- **Variable Naming**: Avoid single-letter variable names except for conventional uses (e.g., i, j, k in loops; e in exception handling).
- **Units in Names**: For variables representing a physical quantity, append the unit to the name. All internal calculations MUST be in SI units.
  - **Example**: object_mass_kg, pipe_diameter_m, signal_duration_s.

### 2.5. Data Handling and Configuration
- **Type Hinting**: Use type hints (as per PEP 484) for all function signatures (arguments and return types) and for complex variable assignments.
- **Data Validation**: Use Pydantic models for defining, validating, and managing complex data structures, especially for external data exchange (e.g., API requests/responses, configuration).
- **Hardcoded Values**:
  - Business logic, environment-dependent, or user-configurable values MUST NOT be hardcoded.
  - Algorithmic constants and protocol-defined values MAY be hardcoded if documented.
  - Use `UPPER_SNAKE_CASE` class-level constants for values that are fixed within the code's logic.
  - Use configuration files (e.g., `config.yaml`, `.env`) for parameters that users or deployment environments might need to change.
- **Default Values**:
  - Defaults SHOULD be minimised justified.
  - Defaults MUST NOT violate Fail Fast (i.e., MUST NOT hide invalid inputs or states).
  - Defaults MUST be documented and MUST represent safe, non-surprising behavior.

- **Secrets Management**: Secrets (API keys, passwords, etc.)
   - MUST NOT be stored in the code. Use environment variables (loaded via a library like `pydantic-settings`) or a dedicated secrets management service.
   - MUST NOT be printed or logged

### 2.6. Error Handling and Logging
- **Error Handling**: Code MUST fail with clear, actionable errors; it MUST NOT silently continue in an invalid state.
- Use specific, custom exceptions where appropriate to provide meaningful error messages. Avoid catching generic `Exception`.
- When re-raising exceptions, use exception chaining (`raise NewError(...) from exc`) to preserve root cause context.
- **Logging**: Implement structured logging using the standard `logging` module.
  - Configure loggers at the module level.
  - Use different log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) appropriately.
  - Log messages SHOULD be clear and provide context (class and method names) to help with debugging.
  - `print()` MUST NOT be used in production code.
  - Use `logging` exclusively.
  - Logging SHOULD be structured (key/value) where helpful (request_id, user_id, job_id, etc.).
  - **Log Output**: In containerized environments, direct all log output to `stdout` and `stderr`. This allows the runtime environment (e.g., Docker, Kubernetes) to handle log collection, aggregation, and routing. For local development or non-containerized deployments, logging to a file is an acceptable alternative, but the primary mechanism SHOULD be stream-based.

### 2.7. Functions, Methods, and Docstrings
- **Type Hinting**:
  - All functions and methods MUST be fully typed.
- **Docstrings**:
  - All modules, classes, functions, and methods MUST have a docstring.
  - Use the Google Python Style for docstrings.
  - A simple function can have a single-line docstring.
  - Complex functions MUST have a detailed docstring, including descriptions of arguments, return values, and any exceptions raised.
  - For Pydantic models, field-level documentation MUST be provided using
  `Field(..., description="...")`.
- **Docstring Maintenance**: When a function is modified, its docstring MUST be updated to reflect the new behavior.

### 2.8. Dependency Management
- **Tooling (Required)**: Projects MUST use `uv` for dependency and environment management.
  - Dependencies MUST be declared in `pyproject.toml`.
  - Projects MUST commit a lock file to ensure deterministic, reproducible builds.
- **Reproducibility**: Use a dependency management tool that supports `pyproject.toml` and generates a lock file to ensure reproducible environments. Pinned versions in a lock file are mandatory for applications.

### 2.9. Refactoring
- **Goals**: When refactoring, prioritize improvements in readability, simplicity (**KISS**), and adherence to the DRY principle.
- **Convention Adherence**: Ensure that any refactored code maintains or improves adherence to all conventions in this document.
- **Test Maintenance**: See Section 4.4 for testing requirements during refactoring, which vary by project phase.

### 2.10. Loop Safety and Defensive Programming

- **Bounded Loops**: `while True` loops MUST have:
  1. A clear exit condition that is guaranteed to eventually trigger
  2. A maximum iteration safety limit as a fail-safe
  3. Type validation on values that determine loop exit

  **Bad Example:**

  ```python
  # DANGEROUS: Infinite loop if bio_read returns truthy non-bytes (e.g., MagicMock)
  while True:
      chunk = ssl_conn.bio_read(8192)
      if not chunk:  # MagicMock is truthy - loop never exits!
          break
      output += chunk
  ```

  **Good Example:**

  ```python
  # SAFE: Bounded iteration with type check
  max_iterations = 1000  # Safety limit
  for _ in range(max_iterations):
      chunk = ssl_conn.bio_read(8192)
      if not chunk or not isinstance(chunk, bytes):
          break
      output += chunk
  ```

- **Prefer `for` over `while True`**: When iteration count is bounded or can be bounded with a safety limit, use `for _ in range(max)` instead of `while True` with a counter.

- **Document Loop Invariants**: Complex loops MUST document what condition guarantees termination in comments or docstrings.

- **Defensive Type Checking**: When processing external or potentially mocked data in loops:
  - Validate types before using values in control flow decisions
  - Use `isinstance()` checks for critical type assumptions
  - This prevents test mocks and malformed data from causing unbounded behavior

### 2.11. Recursion Safety and Depth Limiting

Recursive functions and recursive system calls (e.g., LLM calling LLM) MUST be bounded to prevent stack overflow and runaway resource consumption.

- **Max Depth Parameter**: All recursive functions MUST accept a `max_depth` or `depth` parameter:
  1. Functions MUST track current recursion depth
  2. Functions MUST refuse to recurse beyond `max_depth`
  3. Exceeding max depth MUST raise an exception or return an error, not silently fail

  **Bad Example:**

  ```python
  # DANGEROUS: Unbounded recursion
  def process_nested(data):
      if isinstance(data, dict):
          return {k: process_nested(v) for k, v in data.items()}
      return data
  ```

  **Good Example:**

  ```python
  # SAFE: Bounded recursion with depth tracking
  MAX_RECURSION_DEPTH = 10

  def process_nested(data: Any, *, _depth: int = 0) -> Any:
      """Process nested data structure with bounded recursion.

      Args:
          data: Data to process.
          _depth: Current recursion depth (internal use).

      Raises:
          RecursionError: If max depth exceeded.
      """
      if _depth > MAX_RECURSION_DEPTH:
          raise RecursionError(f"Max recursion depth {MAX_RECURSION_DEPTH} exceeded")

      if isinstance(data, dict):
          return {k: process_nested(v, _depth=_depth + 1) for k, v in data.items()}
      return data
  ```

- **Recursive LLM/Agent Calls**: Systems making recursive LLM calls (sub-LM querying sub-LM) MUST:
  1. Track recursion depth in session/context state
  2. Enforce configurable `max_depth` (default: 1 for sub-calls using base LLM)
  3. Prevent `llm_query()` access at max depth level
  4. Log recursion depth for debugging and cost tracking

  **Example Pattern:**

  ```python
  class LMHandler:
      """Mediates recursive LLM calls with depth limiting."""

      def __init__(self, max_depth: int = 1) -> None:
          self.max_depth = max_depth
          self._current_depth = 0

      def llm_query(self, prompt: str) -> str:
          """Execute sub-LM query with recursion guard.

          Raises:
              RecursionError: If max_depth would be exceeded.
          """
          if self._current_depth >= self.max_depth:
              raise RecursionError(
                  f"Max LLM recursion depth {self.max_depth} reached. "
                  "Sub-LM calls not available at this depth."
              )

          self._current_depth += 1
          try:
              return self._execute_llm_call(prompt)
          finally:
              self._current_depth -= 1
  ```

- **Rate Limiting for Recursive Calls**: Recursive external calls MUST be rate-limited:
  1. Track total call count per session
  2. Enforce `max_calls_per_session` limit
  3. Return clear error when limit reached

- **Document Recursion Contracts**: Functions that recurse or spawn recursive calls MUST document:
  1. Maximum expected depth in docstring
  2. What happens when max depth is reached
  3. Any state that accumulates across recursive calls


## 3. Automation with Pre-Commit Hooks
To automate adherence to these conventions, the project MUST use pre-commit hooks. A `.pre-commit-config.yaml` file SHOULD be configured in the repository root.

### 3.1. Recommended Hooks
- Prefer `ruff` for linting + import sorting
- use `black` for formatting
- `mypy`: For static type checking to enforce the type hints required by section 2.5.
- avoid overlapping tools unless necessary.

## 4. Testing Conventions

### 4.1. Testing Strategy by Project Phase

The testing approach varies based on the project phase to balance code quality with development velocity.

**Project Phase Definitions:**

- **Pre-MVP Phase:** Initial prototyping and feature development until core features are functional
  - Transition criteria: Core workflows functional, basic security working, services start reliably

- **MVP Phase:** Feature-complete for basic operations, suitable for controlled deployment
  - Transition criteria: All requirements implemented, performance targets met, disaster recovery tested, user documentation complete

- **V1.0 Phase:** Production-ready with comprehensive testing, full documentation, and production hardening

**Testing Requirements by Phase:**

#### Pre-MVP Phase (Initial Development)
- **Selective automated testing required**
- Focus on rapid prototyping and feature development
- Manual testing and validation is acceptable
- Code quality conventions (type hints, docstrings, error handling) still apply
- Pre-MVP: automated tests REQUIRED for the listed critical categories; otherwise optional:
  - Safety-critical code
  - Security-sensitive logic
  - Data-destructive operations

#### MVP Phase (Minimum Viable Product)
- **Selective testing** of key functionality:
  - Core business logic functions
  - Critical security features (VPN binding, secrets management)
  - Data integrity operations (backup/restore functions)
  - Configuration validation logic
- Tests SHOULD be added only for components that are feature-complete and stable
- Test coverage targets: 40-60% for critical paths only

#### V1.0 Phase (Production Release)
- **Comprehensive testing required**:
  - Unit tests for all modules, classes, and functions
  - Integration tests for service interactions
  - End-to-end tests for critical user workflows
  - Test coverage target: 80%+ overall, 100% for critical paths
- All tests MUST pass before release
- Implement continuous testing in CI/CD pipeline

### 4.2. Testing Framework and Tools
When testing is required:
- Use `pytest` as the primary testing framework
- Use `pytest-cov` for coverage reporting
- Use `pytest-mock` for mocking external dependencies
- Store tests in a `tests/` directory mirroring the source structure

### 4.3. Test Quality Standards

When tests are written:

- Tests MUST be clear, focused, and test one thing at a time
- Use descriptive test names that explain what is being tested
- Follow the Arrange-Act-Assert (AAA) pattern
- Mock external dependencies (filesystem, network, databases)
- Tests SHOULD be fast and deterministic (no flaky tests)

- **Mock Configuration Requirements**:
  - When mocking objects, ALL methods called by production code MUST be explicitly configured
  - `MagicMock()` returns truthy values by default - this can cause infinite loops in production code
  - Methods used in loop exit conditions MUST return appropriate falsy/empty values

  **Bad Example:**

  ```python
  # DANGEROUS: bio_read() returns MagicMock (truthy), causes infinite loop
  mock_ssl = MagicMock()
  server._sessions[addr] = DTLSSession(ssl_connection=mock_ssl, ...)
  await server.stop()  # Hangs forever in _read_bio_output
  ```

  **Good Example:**

  ```python
  # SAFE: Explicitly configure methods that affect control flow
  mock_ssl = MagicMock()
  mock_ssl.bio_read.return_value = b""  # Returns empty bytes (falsy)
  mock_ssl.shutdown.return_value = None
  ```

- **Mock Audit Checklist**: Before committing tests with mocks, verify:
  1. What methods does production code call on this mock?
  2. Are any of those methods used in loop conditions or control flow?
  3. Are return values configured to allow normal code execution?

### 4.4. Refactoring and Test Maintenance
- When code is modified in MVP or V1.0 phases, corresponding tests MUST be updated
- When bugs are discovered in MVP or V1.0 phases, write a test that reproduces the bug before fixing it (test-driven bug fixing)

### 4.5 Test Naming Convention
- Test files MUST be named `test_<module>.py`
- Test functions MUST start with `test_`

### 4.6 Smoke Tests

Smoke tests are quick verification tests that demonstrate feature functionality with visible output. They complement unit tests by providing human-readable verification of feature behavior.

**Location:** `scripts/smoke_tests.py`

**Run with:** `uv run python scripts/smoke_tests.py`

**Requirements:**
- When adding a new feature, a corresponding smoke test MUST be added
- Smoke tests MUST print clear output showing what is being verified
- Smoke tests MUST return explicit pass/fail status
- Smoke tests SHOULD exercise the feature's public API
- Smoke tests SHOULD include realistic example inputs

**Smoke Test Structure:**
```python
def test_feature_name() -> bool:
    """Test feature description (REQ-ID)."""
    print("=" * 60)
    print("Test N: Feature Name")
    print("=" * 60)

    # Setup
    # ...

    # Exercise feature
    # ...

    # Verify and print results
    if not expected_condition:
        print("✗ FAILED: Reason")
        return False
    print("✓ Verification passed")

    print("\n✓ Feature name test PASSED")
    return True
```

**Purpose:**
- Provide quick sanity check after changes
- Document expected feature behavior through executable examples
- Enable visual verification of output formats
- Supplement unit tests with integration-style checks

---

## 5. Requirements Traceability
To ensure strict compliance with the project's specifications, all code and tests MUST be traceable back to specific Requirement IDs.

### 5.1. Requirement ID Structure
Code and documentation MUST adhere to the ID format defined in the `project_requirement.md`. Deviation constitutes a validation failure.
- High-Level (STK/SYS): <LEVEL>-<TYPE>-<AREA>-<NNNNN> Example: SYS-FUN-NET-0001
- Component-Level (CMP/IMPL): <LEVEL>-<comp_name>-<NNNNN> Example: IMPL-iInterface-0020

### 5.2. Code Implementation Linking
Any class, method, or function that implements a specific requirement MUST cite the ID in its docstring.
- Syntax: Use the tag @req: <ID> within the docstring.
- Scope:
   - STK/SYS IDs SHOULD appear in module-level docstrings or high-level Architecture classes.
   - CMP/IMPL IDs MUST appear in the specific functions or classes performing the logic.

### 5.3. Test Traceability
Tests act as the verification layer for requirements.
- Mapping: Every test case MUST reference the Requirement ID it verifies.
- Syntax: Use the pytest.mark decorator with a custom req marker or include the @req: <ID> tag in the test function docstring.
- Strictness: A requirement is considered "Verified" only if a passing test exists that references its specific ID.
- Example:
```Python

import pytest

@pytest.mark.req("SYS-FUN-NET-0001")
def test_network_latency_compliance():
    ...
```

## 6. Issue Resolution
- When an error arises:
  - Carefully read the logs, stack traces, and error messages
  - Do not stop on the first plausible cause. Dig deep and identify the true root cause.
  - Generate a plan to sort out the root cause
  - Propose the plan for review before implementation

---

## 7. AI Assistant Expectations
- AI assistants MUST follow all MUST-level rules.
- When uncertain, AI assistants SHOULD ask for clarification.
- AI-generated code MUST NOT invent requirements, defaults, or configuration values.
- AI assistants SHOULD prefer explicitness over cleverness.

## 8. Non-Goals
These conventions do NOT:
- Define project-specific architectures or requirements
- Replace project requirement documents
- Mandate specific cloud providers, databases, or frameworks
- Prescribe UI, API, or domain-specific design patterns


---

**End of Conventions**