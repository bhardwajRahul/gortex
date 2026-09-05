package lsp

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// discardWriteCloser is a no-op transport write-end so closeDocument's
// didClose notification can be "sent" without a live LSP server.
type discardWriteCloser struct{}

func (discardWriteCloser) Write(p []byte) (int, error) { return len(p), nil }
func (discardWriteCloser) Close() error                { return nil }

// TestOpenDocument_ConcurrentIsRaceFree reproduces the #714 crash shape:
// concurrent tool requests (e.g. relations.callers → ConfirmSymbolRefs →
// EnsureFileOpen) open overlapping file sets on one provider. On the
// unfixed code the unguarded sourceCache writes are a fatal concurrent
// map writes error under -race; with the docMu claim, every file is
// didOpen'd exactly once and the cache holds one entry per file.
func TestOpenDocument_ConcurrentIsRaceFree(t *testing.T) {
	const files = 12
	repoRoot := t.TempDir()
	for i := 0; i < files; i++ {
		require.NoError(t, os.WriteFile(filepath.Join(repoRoot, fmt.Sprintf("f%d.go", i)), []byte(fmt.Sprintf("package main\n\nfunc F%d() {}\n", i)), 0o644))
	}

	server := newInstrumentedServer()
	p, cleanup := providerWithInstrumentedServer(t, server, []string{"go"}, 4)
	defer cleanup()

	// 8 goroutines each sweep every file — same-file opens race the
	// check-then-act, cross-file opens race the map writes.
	const workers = 8
	var wg sync.WaitGroup
	errs := make(chan error, workers*files)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < files; i++ {
				if err := p.EnsureFileOpen(repoRoot, fmt.Sprintf("f%d.go", i)); err != nil {
					errs <- err
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("EnsureFileOpen: %v", err)
	}

	require.Eventually(t, func() bool {
		_, opens, _ := server.stats()
		return opens == files
	}, 3*time.Second, 5*time.Millisecond, "each of the %d files must be didOpen'd exactly once across %d concurrent sweepers", files, workers)

	p.docMu.RLock()
	cacheLen := len(p.sourceCache)
	p.docMu.RUnlock()
	assert.Equal(t, files, cacheLen, "one sourceCache entry per opened file")

	// The cached bytes must be what concurrent readers see through the
	// public getter.
	for i := 0; i < files; i++ {
		assert.Contains(t, string(p.getSource(repoRoot, fmt.Sprintf("f%d.go", i))), fmt.Sprintf("func F%d()", i))
	}
}

// TestOpenDocument_FailedOpenLeavesNoClaim pins the failure half of the
// claim: a read or didOpen error must un-claim the path, or a retry
// after the file appears silently skips its didOpen (#714's suggested
// fix closes the check-then-act both ways).
func TestOpenDocument_FailedOpenLeavesNoClaim(t *testing.T) {
	repoRoot := t.TempDir()

	server := newInstrumentedServer()
	p, cleanup := providerWithInstrumentedServer(t, server, []string{"go"}, 4)
	defer cleanup()

	missing := filepath.Join(repoRoot, "gone.go")
	require.Error(t, p.EnsureFileOpen(repoRoot, "gone.go"))

	p.docMu.RLock()
	openClaimed := p.openDocs[missing]
	cacheLen := len(p.sourceCache)
	p.docMu.RUnlock()
	assert.False(t, openClaimed, "a failed open must not leave the path claimed open")
	assert.Zero(t, cacheLen, "a failed open must not leave cached bytes")

	// The retry, once the file exists, must send its didOpen.
	require.NoError(t, os.WriteFile(missing, []byte("package main\n"), 0o644))
	require.NoError(t, p.EnsureFileOpen(repoRoot, "gone.go"))
	require.Eventually(t, func() bool {
		_, opens, _ := server.stats()
		return opens == 1
	}, 3*time.Second, 5*time.Millisecond, "retry after a failed open performs the didOpen")
}

// TestCloseDocument_DropsSourceCache asserts closeDocument frees the
// file's cached bytes. Without this an interactive session that keeps
// the provider warm (hover / definition traffic) retains every
// navigated file's contents for the daemon's lifetime.
func TestCloseDocument_DropsSourceCache(t *testing.T) {
	const abs = "/work/main.go"
	p := &Provider{
		openDocs:    map[string]bool{abs: true},
		docVersions: map[string]int{abs: 1},
		lastDiag:    map[string][]Diagnostic{},
		sourceCache: map[string][]byte{abs: []byte("package main")},
		client:      &Client{stdin: discardWriteCloser{}},
	}

	if err := p.closeDocument(abs); err != nil {
		t.Fatalf("closeDocument: %v", err)
	}
	if _, ok := p.sourceCache[abs]; ok {
		t.Fatalf("sourceCache still holds %q after closeDocument", abs)
	}
	if p.openDocs[abs] {
		t.Fatalf("openDocs still marks %q open after closeDocument", abs)
	}
}

// TestResetForReconnect_ClearsSourceCache asserts a reconnect frees
// every cached file's bytes, not just the doc-version / open-doc
// bookkeeping. A nil client makes resetForReconnect skip the LSP
// shutdown handshake.
func TestResetForReconnect_ClearsSourceCache(t *testing.T) {
	p := &Provider{
		openDocs:    map[string]bool{"/a": true},
		docVersions: map[string]int{"/a": 1},
		lastDiag:    map[string][]Diagnostic{"/a": nil},
		sourceCache: map[string][]byte{"/a": []byte("x"), "/b": []byte("y")},
	}

	p.resetForReconnect()

	if len(p.sourceCache) != 0 {
		t.Fatalf("sourceCache = %d entries after resetForReconnect, want 0", len(p.sourceCache))
	}
	if len(p.openDocs) != 0 {
		t.Fatalf("openDocs = %d entries after resetForReconnect, want 0", len(p.openDocs))
	}
}
