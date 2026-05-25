import math
from collections import defaultdict

from heuristic.physics import *


# ============================================================
# World Model
# ============================================================

def fleet_target_planet(fleet, planets):
    best_planet = None
    best_time = 1e9
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)

    for planet in planets:
        dx = planet.x - fleet.x
        dy = planet.y - fleet.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        radius_sq = planet.radius * planet.radius
        if perp_sq >= radius_sq:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))
        turns = hit_d / speed
        if turns <= HORIZON and turns < best_time:
            best_time = turns
            best_planet = planet

    if best_planet is None:
        return None, None
    return best_planet, int(math.ceil(best_time))


def build_arrival_ledger(fleets, planets):
    arrivals_by_planet = {planet.id: [] for planet in planets}
    for fleet in fleets:
        target, eta = fleet_target_planet(fleet, planets)
        if target is None:
            continue
        arrivals_by_planet[target.id].append((eta, fleet.owner, int(fleet.ships)))
    return arrivals_by_planet


def resolve_arrival_event(owner, garrison, arrivals):
    by_owner = {}
    for _, attacker_owner, ships in arrivals:
        by_owner[attacker_owner] = by_owner.get(attacker_owner, 0) + ships

    if not by_owner:
        return owner, max(0.0, garrison)

    sorted_players = sorted(by_owner.items(), key=lambda item: item[1], reverse=True)
    top_owner, top_ships = sorted_players[0]

    if len(sorted_players) > 1:
        second_ships = sorted_players[1][1]
        if top_ships == second_ships:
            survivor_owner = -1
            survivor_ships = 0
        else:
            survivor_owner = top_owner
            survivor_ships = top_ships - second_ships
    else:
        survivor_owner = top_owner
        survivor_ships = top_ships

    if survivor_ships <= 0:
        return owner, max(0.0, garrison)

    if owner == survivor_owner:
        return owner, garrison + survivor_ships

    garrison -= survivor_ships
    if garrison < 0:
        return survivor_owner, -garrison
    return owner, garrison


def normalize_arrivals(arrivals, horizon):
    events = []
    for turns, owner, ships in arrivals:
        if ships <= 0:
            continue
        eta = max(1, int(math.ceil(turns)))
        if eta > horizon:
            continue
        events.append((eta, owner, int(ships)))
    events.sort(key=lambda item: item[0])
    return events


def simulate_planet_timeline(planet, arrivals, player, horizon):
    horizon = max(0, int(math.ceil(horizon)))
    events = normalize_arrivals(arrivals, horizon)
    by_turn = defaultdict(list)
    for item in events:
        by_turn[item[0]].append(item)

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    min_owned = garrison if owner == player else 0.0
    first_enemy = None
    fall_turn = None

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production

        group = by_turn.get(turn, [])
        prev_owner = owner
        if group:
            if prev_owner == player and first_enemy is None:
                if any(item[1] not in (-1, player) for item in group):
                    first_enemy = turn
            owner, garrison = resolve_arrival_event(owner, garrison, group)
            if prev_owner == player and owner != player and fall_turn is None:
                fall_turn = turn

        owner_at[turn] = owner
        ships_at[turn] = max(0.0, garrison)
        if owner == player:
            min_owned = min(min_owned, garrison)

    keep_needed = 0
    holds_full = True

    if planet.owner == player:
        def survives_with_keep(keep):
            sim_owner = planet.owner
            sim_garrison = float(keep)
            for turn in range(1, horizon + 1):
                if sim_owner != -1:
                    sim_garrison += planet.production
                group = by_turn.get(turn, [])
                if group:
                    sim_owner, sim_garrison = resolve_arrival_event(sim_owner, sim_garrison, group)
                    if sim_owner != player:
                        return False
            return sim_owner == player

        if survives_with_keep(int(planet.ships)):
            lo, hi = 0, int(planet.ships)
            while lo < hi:
                mid = (lo + hi) // 2
                if survives_with_keep(mid):
                    hi = mid
                else:
                    lo = mid + 1
            keep_needed = lo
        else:
            holds_full = False
            keep_needed = int(planet.ships)

    return {
        "owner_at": owner_at,
        "ships_at": ships_at,
        "keep_needed": keep_needed,
        "min_owned": max(0, int(math.floor(min_owned))) if planet.owner == player else 0,
        "first_enemy": first_enemy,
        "fall_turn": fall_turn,
        "holds_full": holds_full,
        "horizon": horizon,
    }


def state_at_timeline(timeline, arrival_turn):
    turn = max(0, int(math.ceil(arrival_turn)))
    turn = min(turn, timeline["horizon"])
    owner = timeline["owner_at"].get(turn, timeline["owner_at"][timeline["horizon"]])
    ships = timeline["ships_at"].get(turn, timeline["ships_at"][timeline["horizon"]])
    return owner, max(0.0, ships)


def count_players(planets, fleets):
    owners = set()
    for planet in planets:
        if planet.owner != -1:
            owners.add(planet.owner)
    for fleet in fleets:
        owners.add(fleet.owner)
    return max(2, len(owners))


def nearest_distance_to_set(px, py, planets):
    if not planets:
        return 10**9
    return min(dist(px, py, planet.x, planet.y) for planet in planets)


def indirect_features(planet, planets, player):
    friendly = 0.0
    neutral = 0.0
    enemy = 0.0
    for other in planets:
        if other.id == planet.id:
            continue
        d = dist(planet.x, planet.y, other.x, other.y)
        if d < 1:
            continue
        factor = other.production / (d + 12.0)
        if other.owner == player:
            friendly += factor
        elif other.owner == -1:
            neutral += factor
        else:
            enemy += factor
    return friendly, neutral, enemy


def detect_exposed_enemy_planets(fleets, enemy_planets):
    exposed = set()
    for planet in enemy_planets:
        outbound = sum(
            int(f.ships)
            for f in fleets
            if f.owner == planet.owner and f.from_planet_id == planet.id and f.ships >= 5
        )
        if outbound >= 12 and outbound >= planet.ships * 0.8:  # lowered threshold
            exposed.add(planet.id)
    return exposed


def _compute_weakest_enemy(enemy_planets, owner_strength, owner_production):
    """Return the player ID of the weakest enemy (lowest ships + future production)."""
    enemy_owners = set(p.owner for p in enemy_planets)
    if not enemy_owners:
        return None
    return min(
        enemy_owners,
        key=lambda owner: owner_strength.get(owner, 0) + owner_production.get(owner, 0) * 15,
    )


class WorldModel:
    def __init__(self, player, step, planets, fleets, initial_by_id, ang_vel, comets, comet_ids):
        self.player = player
        self.step = step
        self.planets = planets
        self.fleets = fleets
        self.initial_by_id = initial_by_id
        self.ang_vel = ang_vel
        self.comets = comets
        self.comet_ids = set(comet_ids)

        self.planet_by_id = {planet.id: planet for planet in planets}
        self.my_planets = [planet for planet in planets if planet.owner == player]
        self.enemy_planets = [planet for planet in planets if planet.owner not in (-1, player)]
        self.neutral_planets = [planet for planet in planets if planet.owner == -1]
        self.static_neutral_planets = [
            planet for planet in self.neutral_planets if is_static_planet(planet)
        ]

        # Per-opponent grouping for reinforcement analysis (from ykhnkf)
        self.opp_planets = defaultdict(list)
        for p in self.enemy_planets:
            self.opp_planets[p.owner].append(p)

        self.num_players = count_players(planets, fleets)
        self.remaining_steps = max(1, TOTAL_STEPS - step)
        self.is_early = step < EARLY_TURN_LIMIT
        self.is_opening = step < OPENING_TURN_LIMIT
        self.is_late = self.remaining_steps < LATE_REMAINING_TURNS
        self.is_very_late = self.remaining_steps < VERY_LATE_REMAINING_TURNS
        self.is_total_war = self.remaining_steps < TOTAL_WAR_REMAINING_TURNS
        self.is_four_player = self.num_players >= 4

        self.owner_strength = defaultdict(int)
        self.owner_production = defaultdict(int)
        for planet in planets:
            if planet.owner != -1:
                self.owner_strength[planet.owner] += int(planet.ships)
                self.owner_production[planet.owner] += int(planet.production)
        for fleet in fleets:
            self.owner_strength[fleet.owner] += int(fleet.ships)

        self.my_total = self.owner_strength.get(player, 0)
        self.enemy_total = sum(
            strength for owner, strength in self.owner_strength.items() if owner != player
        )
        self.max_enemy_strength = max(
            (strength for owner, strength in self.owner_strength.items() if owner != player),
            default=0,
        )
        self.my_prod = self.owner_production.get(player, 0)
        self.enemy_prod = sum(
            production
            for owner, production in self.owner_production.items()
            if owner != player
        )

        # Weakest enemy tracking (key for 4P elimination strategy)
        self._weakest_enemy = _compute_weakest_enemy(
            self.enemy_planets, self.owner_strength, self.owner_production
        )
        self._weakest_enemy_strength = (
            self.owner_strength.get(self._weakest_enemy, 0)
            if self._weakest_enemy is not None else 0
        )

        self.arrivals_by_planet = build_arrival_ledger(fleets, planets)
        self.base_timeline = {
            planet.id: simulate_planet_timeline(
                planet,
                self.arrivals_by_planet[planet.id],
                player,
                HORIZON,
            )
            for planet in planets
        }
        self.keep_needed_map = {
            planet.id: self.base_timeline[planet.id]["keep_needed"] for planet in planets
        }
        self.min_owned_map = {
            planet.id: self.base_timeline[planet.id]["min_owned"] for planet in planets
        }
        self.first_enemy_map = {
            planet.id: self.base_timeline[planet.id]["first_enemy"] for planet in planets
        }
        self.fall_turn_map = {
            planet.id: self.base_timeline[planet.id]["fall_turn"] for planet in planets
        }
        self.holds_full_map = {
            planet.id: self.base_timeline[planet.id]["holds_full"] for planet in planets
        }
        self.indirect_feature_map = {
            planet.id: indirect_features(planet, planets, player) for planet in planets
        }
        self.exposed_planet_ids = detect_exposed_enemy_planets(fleets, self.enemy_planets)
        self.shot_cache = {}
        self.probe_candidate_cache = {}
        self.best_probe_cache = {}
        self.reaction_cache = {}
        self.exact_need_cache = {}

        self.total_visible_ships = sum(int(planet.ships) for planet in planets) + sum(
            int(fleet.ships) for fleet in fleets
        )
        self.total_production = sum(int(planet.production) for planet in planets)

    def is_static(self, planet_id):
        return is_static_planet(self.planet_by_id[planet_id])

    def comet_life(self, planet_id):
        return comet_remaining_life(planet_id, self.comets)

    def source_inventory_left(self, source_id, spent_total):
        return max(0, int(self.planet_by_id[source_id].ships) - spent_total[source_id])

    def plan_shot(self, src_id, target_id, ships):
        ships = int(ships)
        key = (src_id, target_id, ships)
        if key in self.shot_cache:
            return self.shot_cache[key]
        src = self.planet_by_id[src_id]
        target = self.planet_by_id[target_id]
        result = aim_with_prediction(
            src, target, ships, self.initial_by_id, self.ang_vel, self.comets, self.comet_ids,
        )
        self.shot_cache[key] = result
        return result

    def probe_ship_candidates(self, src_id, target_id, source_cap, hints=()):
        source_cap = max(1, int(source_cap))
        normalized_hints = tuple(
            int(math.ceil(hint)) for hint in hints if hint is not None
        )
        cache_key = (src_id, target_id, source_cap, normalized_hints)
        cached = self.probe_candidate_cache.get(cache_key)
        if cached is not None:
            return cached
        target = self.planet_by_id[target_id]
        target_ships = max(1, int(math.ceil(target.ships)))

        values = set(range(1, min(6, source_cap) + 1))
        values.update({
            source_cap,
            max(1, source_cap // 2),
            max(1, source_cap // 3),
            min(source_cap, PARTIAL_SOURCE_MIN_SHIPS),
            min(source_cap, target_ships + 1),
            min(source_cap, target_ships + 2),
            min(source_cap, target_ships + 4),
            min(source_cap, target_ships + 8),
        })

        for hint in normalized_hints:
            base = max(1, min(source_cap, hint))
            for delta in (-2, -1, 0, 1, 2):
                candidate = base + delta
                if 1 <= candidate <= source_cap:
                    values.add(candidate)

        result = sorted(values)
        self.probe_candidate_cache[cache_key] = result
        return result

    def best_probe_aim(
        self,
        src_id,
        target_id,
        source_cap,
        hints=(),
        min_turn=None,
        max_turn=None,
        anchor_turn=None,
        max_anchor_diff=None,
    ):
        cache_key = (
            src_id,
            target_id,
            max(1, int(source_cap)),
            tuple(hints),
            min_turn,
            max_turn,
            anchor_turn,
            max_anchor_diff,
        )
        if cache_key in self.best_probe_cache:
            return self.best_probe_cache[cache_key]

        best = None
        best_key = None

        for ships in self.probe_ship_candidates(src_id, target_id, source_cap, hints=hints):
            aim = self.plan_shot(src_id, target_id, ships)
            if aim is None:
                continue

            angle, turns, dist_to_target, path_target = aim
            if min_turn is not None and turns < min_turn:
                continue
            if max_turn is not None and turns > max_turn:
                continue
            if (
                anchor_turn is not None
                and max_anchor_diff is not None
                and abs(turns - anchor_turn) > max_anchor_diff
            ):
                continue

            if anchor_turn is None:
                key = (turns, ships)
            else:
                key = (abs(turns - anchor_turn), turns, ships)

            if best_key is None or key < best_key:
                best_key = key
                best = (ships, (angle, turns, dist_to_target, path_target))

        self.best_probe_cache[cache_key] = best
        return best

    def reaction_times(self, target_id):
        cached = self.reaction_cache.get(target_id)
        if cached is not None:
            return cached

        target = self.planet_by_id[target_id]
        my_t = 10**9
        for planet in self.my_planets:
            seeded = self.best_probe_aim(planet.id, target.id, max(1, int(planet.ships)))
            if seeded is None:
                continue
            _, aim = seeded
            my_t = min(my_t, aim[1])

        enemy_t = 10**9
        for planet in self.enemy_planets:
            seeded = self.best_probe_aim(planet.id, target.id, max(1, int(planet.ships)))
            if seeded is None:
                continue
            _, aim = seeded
            enemy_t = min(enemy_t, aim[1])

        cached = (my_t, enemy_t)
        self.reaction_cache[target_id] = cached
        return cached

    def projected_state(self, target_id, arrival_turn, planned_commitments=None, extra_arrivals=()):
        planned_commitments = planned_commitments or {}
        cutoff = max(1, int(math.ceil(arrival_turn)))
        if not planned_commitments.get(target_id) and not extra_arrivals:
            return state_at_timeline(self.base_timeline[target_id], cutoff)

        arrivals = [
            item
            for item in self.arrivals_by_planet.get(target_id, [])
            if item[0] <= cutoff
        ]
        arrivals.extend(
            item
            for item in planned_commitments.get(target_id, [])
            if item[0] <= cutoff
        )
        arrivals.extend(item for item in extra_arrivals if item[0] <= cutoff)

        target = self.planet_by_id[target_id]
        dyn = simulate_planet_timeline(target, arrivals, self.player, cutoff)
        return state_at_timeline(dyn, cutoff)

    def projected_timeline(self, target_id, horizon, planned_commitments=None, extra_arrivals=()):
        planned_commitments = planned_commitments or {}
        horizon = max(1, int(math.ceil(horizon)))
        arrivals = [
            item for item in self.arrivals_by_planet.get(target_id, []) if item[0] <= horizon
        ]
        arrivals.extend(
            item for item in planned_commitments.get(target_id, []) if item[0] <= horizon
        )
        arrivals.extend(item for item in extra_arrivals if item[0] <= horizon)
        target = self.planet_by_id[target_id]
        return simulate_planet_timeline(target, arrivals, self.player, horizon)

    def hold_status(self, target_id, planned_commitments=None, horizon=HORIZON):
        planned_commitments = planned_commitments or {}
        if planned_commitments.get(target_id):
            tl = self.projected_timeline(
                target_id, horizon, planned_commitments=planned_commitments,
            )
        else:
            tl = self.base_timeline[target_id]
        return {
            "keep_needed": tl["keep_needed"],
            "min_owned": tl["min_owned"],
            "first_enemy": tl["first_enemy"],
            "fall_turn": tl["fall_turn"],
            "holds_full": tl["holds_full"],
        }

    def _ownership_search_cap(self, eval_turn):
        productive_cap = self.total_production * max(2, eval_turn + 2)
        return max(32, int(self.total_visible_ships + productive_cap + 32))

    def min_ships_to_own_by(
        self,
        target_id,
        eval_turn,
        attacker_owner,
        arrival_turn=None,
        planned_commitments=None,
        extra_arrivals=(),
        upper_bound=None,
    ):
        planned_commitments = planned_commitments or {}
        eval_turn = max(1, int(math.ceil(eval_turn)))
        arrival_turn = eval_turn if arrival_turn is None else max(1, int(math.ceil(arrival_turn)))
        if arrival_turn > eval_turn:
            if upper_bound is not None:
                return max(1, int(upper_bound)) + 1
            return self._ownership_search_cap(eval_turn) + 1

        normalized_extra = tuple(
            (max(1, int(math.ceil(turns))), owner, int(ships))
            for turns, owner, ships in extra_arrivals
            if ships > 0 and max(1, int(math.ceil(turns))) <= eval_turn
        )

        cache_key = None
        if (
            arrival_turn == eval_turn
            and not planned_commitments.get(target_id)
            and not normalized_extra
        ):
            cache_key = (target_id, eval_turn, attacker_owner)
            cached = self.exact_need_cache.get(cache_key)
            if cached is not None:
                return cached

        owner_before, ships_before = self.projected_state(
            target_id,
            eval_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=normalized_extra,
        )
        if owner_before == attacker_owner:
            if cache_key is not None:
                self.exact_need_cache[cache_key] = 0
            return 0

        def owns_at(ships):
            owner_after, _ = self.projected_state(
                target_id,
                eval_turn,
                planned_commitments=planned_commitments,
                extra_arrivals=normalized_extra + ((arrival_turn, attacker_owner, int(ships)),),
            )
            return owner_after == attacker_owner

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not owns_at(hi):
                return hi + 1
        else:
            hi = max(1, int(math.ceil(ships_before)) + 1)
            search_cap = self._ownership_search_cap(eval_turn)
            while hi <= search_cap and not owns_at(hi):
                hi *= 2
            if hi > search_cap:
                hi = search_cap
                if not owns_at(hi):
                    return hi + 1

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if owns_at(mid):
                hi = mid
            else:
                lo = mid + 1

        if cache_key is not None:
            self.exact_need_cache[cache_key] = lo
        return lo

    def min_ships_to_own_at(
        self,
        target_id,
        arrival_turn,
        attacker_owner,
        planned_commitments=None,
        extra_arrivals=(),
        upper_bound=None,
    ):
        return self.min_ships_to_own_by(
            target_id,
            arrival_turn,
            attacker_owner,
            arrival_turn=arrival_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
            upper_bound=upper_bound,
        )

    def reinforcement_needed_to_hold_until(
        self,
        planet_id,
        arrival_turn,
        hold_until,
        planned_commitments=None,
        upper_bound=None,
    ):
        planned_commitments = planned_commitments or {}
        target = self.planet_by_id[planet_id]
        arrival_turn = max(1, int(math.ceil(arrival_turn)))
        hold_until = max(arrival_turn, int(math.ceil(hold_until)))

        if target.owner != self.player:
            return self.min_ships_to_own_by(
                planet_id,
                hold_until,
                self.player,
                arrival_turn=arrival_turn,
                planned_commitments=planned_commitments,
                upper_bound=upper_bound,
            )

        def holds_with_reinforcement(ships):
            timeline = self.projected_timeline(
                planet_id,
                hold_until,
                planned_commitments=planned_commitments,
                extra_arrivals=((arrival_turn, self.player, int(ships)),),
            )
            for turn in range(arrival_turn, hold_until + 1):
                if timeline["owner_at"].get(turn) != self.player:
                    return False
            return True

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not holds_with_reinforcement(hi):
                return hi + 1
        else:
            hi = 1
            search_cap = self._ownership_search_cap(hold_until)
            while hi <= search_cap and not holds_with_reinforcement(hi):
                hi *= 2
            if hi > search_cap:
                hi = search_cap
                if not holds_with_reinforcement(hi):
                    return hi + 1

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if holds_with_reinforcement(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def ships_needed_to_capture(
        self,
        target_id,
        arrival_turn,
        planned_commitments=None,
        extra_arrivals=(),
    ):
        return self.min_ships_to_own_at(
            target_id,
            arrival_turn,
            self.player,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
        )