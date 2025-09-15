# Changelog

## 0.1.1

- Change: Make `zod` a peer dependency and use type-only imports to avoid schema type mismatch across installations.
- Typing: `DefaultAdapterTask<T>['schema']` now uses `ZodSchema<T>` (type-only) to accept user-provided Zod schemas without requiring the library's instance.
- Typing: `GenerateObjectOptions<T>['schema']` updated to `ZodSchema<T>`.
- Build: Re-export `ZodSchema` type from the package entry for convenience.

## 0.1.0

- Initial release.
