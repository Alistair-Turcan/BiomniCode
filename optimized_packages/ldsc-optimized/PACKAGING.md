# Packaging provenance

- Evaluated input: `ldsc-opt-py3.zip`.
- SHA-256: `4f23a97f2524eddbb376a85708603d351df67a5b10650bb6958adf18491556d1`.
- Executable Python source, dependency files, tests, and license are unchanged from the evaluated upload.
- README updated for this fork's installation and measured numerical agreement. Upstream README retained separately.
- PORTING introduction clarified to avoid a cross-stack bitwise guarantee.
- Ignore rules extended for runtime caches and to retain bundled test fixtures in a newly initialized repository.
- SOURCE_MANIFEST.json records every unchanged Python source checksum.
- Removed generated entries: ["test/simulate_test/ldscore/.ldsc_cache/9a40df01fc85cf66577e4867eafe1e20dd1801d0b14023f2170031a3716b9997.pkl"].
- Packaging checks: Python source syntax, source checksum identity, required repository files, ZIP integrity, and prohibited cache paths. No new scientific benchmark was run for packaging.
