CREATE TABLE IF NOT EXISTS swipes (
  id TEXT PRIMARY KEY,
  session TEXT NOT NULL,
  clip TEXT NOT NULL,
  liked INTEGER NOT NULL,
  created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_swipes_session ON swipes (session, created_at);
CREATE INDEX IF NOT EXISTS idx_swipes_clip ON swipes (clip);
