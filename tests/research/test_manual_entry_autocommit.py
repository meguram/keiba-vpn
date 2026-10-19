import subprocess

from src.research.pedigree import manual_entry_server as me


def _git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _repo(tmp_path):
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
    for k, v in (("user.email", "t@example.test"), ("user.name", "t"), ("commit.gpgsign", "false")):
        _git(work, "config", k, v)
    _git(work, "remote", "add", "origin", str(remote))
    d = work / "meta"
    d.mkdir()
    anc = work / "anc.csv"
    anc.write_text("horse_id,name,birthday,precision,source,source_url,wiki_conflict\n" + "".join(f"h{i},Horse{i},19{60 + i},year,n,u,\n" for i in range(8)), encoding="utf-8")
    zone = work / "zone.csv"
    zone.write_text("stallion_id,stallion_name,stallion_birth_year,conception_year,min_gen,n_horses,manual_value,note\n" + "".join(f"h{i},H,1960,2000,1,{100 - i},,\n" for i in range(8)), encoding="utf-8")
    (d / "birth_manual.csv").write_text(",".join(me.MANUAL_COLUMNS) + "\n", encoding="utf-8")
    (d / "birth_search_log.csv").write_text(",".join(me.LOG_COLUMNS) + "\n", encoding="utf-8")
    (work / "other.txt").write_text("keep out of commit", encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "init")
    _git(work, "push", "-q", "-u", "origin", "main")
    return work, remote, anc, zone, d


def test_auto_commit_every_n_and_only_data_files(tmp_path):
    work, remote, anc, zone, d = _repo(tmp_path)
    (work / "other.txt").write_text("dirty", encoding="utf-8")               # 無関係な未コミット変更は巻き込まない
    s = me.Store(anc, zone, None, d, "https://example.test/p", commit_every=3)
    base = _git(work, "rev-list", "--count", "HEAD")
    assert s.save("h0", "1960-03-04")["ok"] and s.save("h1", "1961-04-05")["ok"]
    assert _git(work, "rev-list", "--count", "HEAD") == base                 # まだ 2 件
    assert s.none("h2")["ok"]                                                # 3 件目でコミット・プッシュ
    assert int(_git(work, "rev-list", "--count", "HEAD")) == int(base) + 1
    assert "プッシュしました" in s.git_status and s.pending_msgs == []
    assert _git(remote, "rev-list", "--count", "main") == _git(work, "rev-list", "--count", "HEAD")
    files = _git(work, "show", "--name-only", "--format=", "HEAD").splitlines()
    assert sorted(files) == ["meta/birth_manual.csv", "meta/birth_search_log.csv"]
    assert _git(work, "status", "--short", "other.txt") == "M other.txt"     # 無関係な変更は未コミットのまま
    assert "Co-Authored-By: Claude Sonnet 5" in _git(work, "log", "-1", "--format=%B")


def test_commit_now_and_disabled(tmp_path):
    work, remote, anc, zone, d = _repo(tmp_path)
    s = me.Store(anc, zone, None, d, "https://example.test/p", commit_every=5)
    s.save("h0", "1960-03-04")
    assert s.next()["unsaved"] == 1
    assert s.commit_now()["ok"] and s.pending_msgs == []
    s2 = me.Store(anc, zone, None, d, "https://example.test/p", commit_every=0)
    assert s2.commit_now()["ok"] is False


def test_startup_picks_up_uncommitted_leftovers(tmp_path):
    work, remote, anc, zone, d = _repo(tmp_path)
    with open(d / "birth_manual.csv", "a", encoding="utf-8") as f:
        f.write("hx,X,1970,1,2,u,note\n")                                  # 前回の保存が未コミットで残っている
    s = me.Store(anc, zone, None, d, "https://example.test/p", commit_every=5)
    assert s.next()["unsaved"] == 1
    assert s.commit_now()["ok"] and "プッシュしました" in s.git_status
    assert "hx,X,1970" in _git(work, "show", "HEAD:meta/birth_manual.csv")
