# Changelog

## 0.1.4

- Feature: add `reflectionHint` to steer reflection without templates; interpolated into the default reflection prompt.
- Change: remove `reflectionPromptBuilder` hook to keep the surface minimal.
- Change: simplify `taskLM` wiring by spreading passthrough options into the adapter; all extra fields (e.g., `tools`, `providerOptions`, `stopWhen`, `toolChoice`, `maxToolRoundtrips`, `experimentalTelemetry`) are forwarded to AI SDK calls.
- Docs: update README quickstart to show object-form `taskLM` and `reflectionHint`; add sections on AI SDK passthrough and reflection steering.

## 0.1.2

- Chore: tiny version bump.

## 0.1.1

- Change: Make `zod` a peer dependency and use type-only imports to avoid schema type mismatch across installations.
- Typing: `DefaultAdapterTask<T>['schema']` now uses `ZodSchema<T>` (type-only) to accept user-provided Zod schemas without requiring the library's instance.
- Typing: `GenerateObjectOptions<T>['schema']` updated to `ZodSchema<T>`.
- Build: Re-export `ZodSchema` type from the package entry for convenience.

## 0.1.0

- Initial release.
