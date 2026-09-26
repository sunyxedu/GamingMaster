/*
 * Differential-test driver: the gmrules skat rules (data_processing/rs/gmrules/src/games/
 * skat.rs) against the rule engine of the International Skat Server itself — the
 * SimpleState / GameDeclaration / GameResult classes of the ISS client package
 * (https://skatgame.net/iss/clients.tgz, (c) Michael Buro, Jeff Long, Jan Schaefer,
 * Nathan Taylor, GPLv3; the server shares these classes). The package is NOT part of
 * this repository: scripts/reference/iss_skat_dump.py downloads it into a cache,
 * compiles its rule classes with this file and runs it. This file only CALLS them.
 *
 *   java IssSkatRef play <games> <seed>   random playouts, every move chosen among the
 *                                         moves the ISS engine accepts
 *   java IssSkatRef replay <file>         ISS record lines (archive format, MV[...])
 *                                         replayed move by move
 *
 * One JSON object per game on stdout:
 *   {"id": "...", "record": "<ISS record line; R[...] = the engine's GameResult>",
 *    "steps": [{"seat": s, "phase": "bid|answer|skat_or_hand|discard_and_decl|discards|cardplay",
 *               "max_bid": b, "legal": [...], "move": "..."}, ...],
 *    "finished": true|false, "result": "<GameResult.toString()>", "money": [m0, m1, m2],
 *    "error": null|"..."}
 * "legal" is probed black-box: every candidate move is tried on a copy of the state
 * (SimpleState.makeMove returns null when the engine accepts a move). Candidates: pass +
 * every bid-list value; y / p; the pickup and every hand declaration; after the pickup
 * every game (and some hand / ouvert-trump games that must be refused) with every pair
 * of cards; every card. After the pickup "legal" is {"games": [...], "discards":
 * [[c1, c2], ...]} (c1 < c2, card ids suit * 8 + rank: the accepted set is checked to be
 * exactly games x discards).
 */

import common.GameResult;
import common.SimpleState;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class IssSkatRef {

  static final String SUITS = "DHSC";
  static final String RANKS = "789TJQKA";
  static final int WORLD = SimpleState.WORLD_MOVE;
  static final String[] HAND_DECLS = {
    "DH", "DHS", "DHZ", "DO", "HH", "HHS", "HHZ", "HO", "SH", "SHS", "SHZ", "SO",
    "CH", "CHS", "CHZ", "CO", "GH", "GHS", "GHZ", "GO", "NH", "NOH",
    // not hand games: must be refused before the pickup
    "D", "G", "N", "NO", "GS"};
  static final String[] PICKUP_DECLS = {
    "D", "H", "S", "C", "G", "N", "NO",
    // hand / ouvert trump games: must be refused after the pickup
    "DH", "GH", "NH", "NOH", "GO", "CHS", "SHZ"};

  static String card(int id) {
    return "" + SUITS.charAt(id / 8) + RANKS.charAt(id % 8);
  }

  /** Cards of a bit mask (bit = suit * 8 + rank), ascending, '.'-separated. */
  static String cards(int mask) {
    StringBuilder sb = new StringBuilder();
    for (int c = 0; c < 32; c++) {
      if ((mask & (1 << c)) != 0) {
        if (sb.length() > 0) sb.append('.');
        sb.append(card(c));
      }
    }
    return sb.toString();
  }

  static boolean accepts(SimpleState st, int player, String move) {
    SimpleState c = SimpleState.createCopy(st);
    return c.makeMove(player, move, null) == null;
  }

  static String phaseName(SimpleState st) {
    switch (st.getPhase()) {
      case SimpleState.BID: return "bid";
      case SimpleState.ANSWER: return "answer";
      case SimpleState.SKAT_OR_HAND_DECL: return "skat_or_hand";
      case SimpleState.DISCARD_AND_DECL: return st.getDiscardHalfPhase() ? "discards" : "discard_and_decl";
      case SimpleState.CARDPLAY: return "cardplay";
      default: return "phase" + st.getPhase();
    }
  }

  /** A hand declaration as sent: ouvert games show the declarer's cards. */
  static String handMove(SimpleState st, int p, String decl) {
    return decl.contains("O") ? decl + "." + cards(st.getHand(p)) : decl;
  }

  static String pickupMove(SimpleState st, int p, String game, int a, int b) {
    String m = game + "." + card(a) + "." + card(b);
    if (game.contains("O")) {
      int twelve = st.getHand(p) | st.getSkat();
      m += "." + cards(twelve & ~(1 << a) & ~(1 << b));
    }
    return m;
  }

  /** The legal moves of the player to move. */
  static class Legal {
    List<String> moves = new ArrayList<>();      // bid / answer / skat_or_hand / cardplay
    List<String> games = new ArrayList<>();      // discard_and_decl
    List<int[]> pairs = new ArrayList<>();       // discard_and_decl / discards
    String error;                                // the accepted set is no product

    String json() {
      if (games.isEmpty() && pairs.isEmpty()) return strList(moves);
      StringBuilder sb = new StringBuilder("{\"games\": ").append(strList(games)).append(", \"discards\": [");
      for (int i = 0; i < pairs.size(); i++) {
        if (i > 0) sb.append(", ");
        sb.append('[').append(pairs.get(i)[0]).append(", ").append(pairs.get(i)[1]).append(']');
      }
      return sb.append("]}").toString();
    }
  }

  static Legal legal(SimpleState st, int p) {
    Legal l = new Legal();
    switch (st.getPhase()) {
      case SimpleState.BID:
        if (accepts(st, p, "p")) l.moves.add("p");
        for (int b : SimpleState.BIDS) {
          if (accepts(st, p, "" + b)) l.moves.add("" + b);
        }
        break;
      case SimpleState.ANSWER:
        for (String m : new String[] {"y", "p", "18"}) {
          if (accepts(st, p, m)) l.moves.add(m);
        }
        break;
      case SimpleState.SKAT_OR_HAND_DECL:
        if (accepts(st, p, "s")) l.moves.add("s");
        for (String d : HAND_DECLS) {
          if (accepts(st, p, handMove(st, p, d))) l.moves.add(d);
        }
        break;
      case SimpleState.DISCARD_AND_DECL:
        if (st.getDiscardHalfPhase()) {
          // after a half-move declaration: the discards alone
          for (int a = 0; a < 32; a++) {
            for (int b = a; b < 32; b++) {
              if (accepts(st, p, card(a) + "." + card(b))) l.pairs.add(new int[] {a, b});
            }
          }
          break;
        }
        boolean[][][] ok = new boolean[PICKUP_DECLS.length][32][32];
        boolean[] gameOk = new boolean[PICKUP_DECLS.length];
        boolean[][] pairOk = new boolean[32][32];
        for (int g = 0; g < PICKUP_DECLS.length; g++) {
          for (int a = 0; a < 32; a++) {
            for (int b = a; b < 32; b++) {
              if (accepts(st, p, pickupMove(st, p, PICKUP_DECLS[g], a, b))) {
                ok[g][a][b] = gameOk[g] = pairOk[a][b] = true;
              }
            }
          }
        }
        for (int g = 0; g < PICKUP_DECLS.length; g++) {
          if (gameOk[g]) l.games.add(PICKUP_DECLS[g]);
        }
        for (int a = 0; a < 32; a++) {
          for (int b = a; b < 32; b++) {
            if (pairOk[a][b]) l.pairs.add(new int[] {a, b});
          }
        }
        for (int g = 0; g < PICKUP_DECLS.length; g++) {
          for (int a = 0; a < 32; a++) {
            for (int b = a; b < 32; b++) {
              if (ok[g][a][b] != (gameOk[g] && pairOk[a][b])) {
                l.error = "the accepted declarations x discards are no product";
              }
            }
          }
        }
        break;
      case SimpleState.CARDPLAY:
        for (int c = 0; c < 32; c++) {
          if (accepts(st, p, card(c))) l.moves.add(card(c));
        }
        break;
      default:
        l.error = "no decision in phase " + st.getPhase();
    }
    return l;
  }

  /** The random playout policy: a move among the legal ones. */
  static String choose(SimpleState st, int p, Legal l, Random rng) {
    switch (st.getPhase()) {
      case SimpleState.BID: {
        List<String> bids = new ArrayList<>(l.moves);
        boolean canPass = bids.remove("p");
        double u = rng.nextDouble();
        if ((canPass && u < 0.45) || bids.isEmpty()) return "p";
        if (u < 0.87) return bids.get(0);
        return bids.get(rng.nextInt(Math.min(bids.size(), 25)));       // a jump bid
      }
      case SimpleState.ANSWER:
        return rng.nextDouble() < 0.6 ? "y" : "p";
      case SimpleState.SKAT_OR_HAND_DECL: {
        List<String> hand = new ArrayList<>(l.moves);
        hand.remove("s");
        if (rng.nextDouble() < 0.5 || hand.isEmpty()) return "s";
        return handMove(st, p, hand.get(rng.nextInt(hand.size())));
      }
      case SimpleState.DISCARD_AND_DECL: {
        int[] pr = l.pairs.get(rng.nextInt(l.pairs.size()));
        String g = l.games.get(rng.nextInt(l.games.size()));
        if (rng.nextBoolean()) pr = new int[] {pr[1], pr[0]};                // either order
        return pickupMove(st, p, g, pr[0], pr[1]);
      }
      default:
        return l.moves.get(rng.nextInt(l.moves.size()));
    }
  }

  // ---- output ---------------------------------------------------------------------

  static String esc(String s) {
    return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
  }

  static String strList(List<String> xs) {
    StringBuilder sb = new StringBuilder("[");
    for (int i = 0; i < xs.size(); i++) {
      if (i > 0) sb.append(", ");
      sb.append(esc(xs.get(i)));
    }
    return sb.append(']').toString();
  }

  static class Game {
    String id;
    SimpleState st = new SimpleState(SimpleState.WORLD_VIEW);
    List<String> mv = new ArrayList<>();
    List<String> steps = new ArrayList<>();
    String error;

    Game(String id) {
      this.id = id;
    }

    boolean world(String move) {
      String e = st.makeMove(WORLD, move, null);
      if (e != null) {
        error = "world move " + move + " refused: " + e;
        return false;
      }
      mv.add("w " + move);
      return true;
    }

    /** A decision of seat p (its legal set recorded first). */
    boolean decide(int p, String move, Legal l) {
      if (l.error != null && error == null) error = l.error;
      steps.add("{\"seat\": " + p + ", \"phase\": " + esc(phaseName(st)) + ", \"max_bid\": " + st.getMaxBid()
                + ", \"legal\": " + l.json() + ", \"move\": " + esc(move) + "}");
      String e = st.makeMove(p, move, null);
      if (e != null) {
        error = "move " + p + " " + move + " refused: " + e;
        return false;
      }
      mv.add(p + " " + move);
      return true;
    }

    /** RE / SC: ISS-only moves, not decisions of the game. */
    boolean other(int p, String move) {
      String e = st.makeMove(p, move, null);
      if (e != null) {
        error = "move " + p + " " + move + " refused: " + e;
        return false;
      }
      mv.add(p + " " + move);
      return true;
    }

    String json() {
      boolean fin = st.isFinished();
      GameResult gr = new GameResult();
      int[] money = new int[3];
      if (fin) st.tournamentScoresMoney(money, gr);
      String res = fin ? gr.toString() : "unknown";
      String record = "(;GM[Skat]PC[ISS reference engine]CO[]SE[0]ID[" + id
          + "]DT[2000-01-01/00:00:00/UTC]P0[a]P1[b]P2[c]R0[]R1[]R2[]MV[" + String.join(" ", mv)
          + " ]R[" + res + "] ;)";
      return "{\"id\": " + esc(id) + ", \"record\": " + esc(record) + ", \"steps\": [" + String.join(", ", steps)
          + "], \"finished\": " + fin + ", \"result\": " + esc(res) + ", \"money\": [" + money[0] + ", "
          + money[1] + ", " + money[2] + "], \"error\": " + (error == null ? "null" : esc(error)) + "}";
    }
  }

  static void play(int n, long seed) {
    Random rng = new Random(seed);
    for (int k = 0; k < n; k++) {
      Game g = new Game(seed + "-" + k);
      g.world(g.st.generateWorldMove(rng));                // the deal
      while (!g.st.isFinished() && g.error == null) {
        if (g.st.getPhase() == SimpleState.GET_SKAT) {
          g.world(g.st.generateWorldMove(rng));            // the server shows the skat
          continue;
        }
        int p = g.st.getToMove();
        Legal l = legal(g.st, p);
        if (l.moves.isEmpty() && l.pairs.isEmpty()) {
          g.error = "no legal move in phase " + g.st.getPhase();
          break;
        }
        g.decide(p, choose(g.st, p, l, rng), l);
      }
      System.out.println(g.json());
    }
  }

  static void replay(String path) throws IOException {
    try (BufferedReader br = new BufferedReader(new FileReader(path))) {
      String line;
      int ln = 0;
      while ((line = br.readLine()) != null) {
        ln++;
        int a = line.indexOf("MV[");
        if (a < 0) continue;
        int b = line.indexOf(']', a);
        int i0 = line.indexOf("ID[");
        String id = i0 >= 0 ? line.substring(i0 + 3, line.indexOf(']', i0)) : "line" + ln;
        String[] t = line.substring(a + 3, b).trim().split("\\s+");
        Game g = new Game(id);
        for (int k = 0; k + 1 < t.length && g.error == null; k += 2) {
          String who = t[k], move = t[k + 1];
          if (who.equals("w")) {
            g.world(move);
          } else {
            int p = Integer.parseInt(who);
            if (move.equals("RE") || move.startsWith("SC")) {
              g.other(p, move);
            } else {
              g.decide(p, move, legal(g.st, p));
            }
          }
        }
        System.out.println(g.json());
      }
    }
  }

  public static void main(String[] args) throws IOException {
    if (args.length == 3 && args[0].equals("play")) {
      play(Integer.parseInt(args[1]), Long.parseLong(args[2]));
    } else if (args.length == 2 && args[0].equals("replay")) {
      replay(args[1]);
    } else {
      System.err.println("usage: IssSkatRef play <games> <seed> | replay <file>");
      System.exit(2);
    }
  }
}
