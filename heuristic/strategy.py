import time
import math
from collections import defaultdict

from heuristic.config import *
from heuristic.types import Mission, ShotOption
from heuristic.world_model import nearest_distance_to_set

# ============================================================
# Strategy
# ============================================================

def planet_distance(first, second):
    return math.hypot(first.x - second.x, first.y - second.y)


def nearest_sources_to_target(target, sources, top_k):
    if top_k <= 0 or len(sources) <= top_k:
        return sources
    return sorted(
        sources,
        key=lambda src: (planet_distance(src, target), -int(src.ships), src.id),
    )[:top_k]


def min_legal_reaction_time(target, sources, world):
    best = 10**9
    for src in sources:
        seeded = world.best_probe_aim(src.id, target.id, max(1, int(src.ships)))
        if seeded is None:
            continue
        _, aim = seeded
        best = min(best, aim[1])
    return best


def policy_reaction_times(target_id, policy):
    return policy["reaction_time_map"].get(target_id, (10**9, 10**9))


def candidate_time_valid(target, turns, world, remaining_buffer):
    if turns > world.remaining_steps - remaining_buffer:
        return False
    if target.id in world.comet_ids:
        life = world.comet_life(target.id)
        if turns >= life or turns > COMET_MAX_CHASE_TURNS:
            return False
    return True


def stacked_enemy_proactive_keep(planet, world):
    threats = []
    for enemy in world.enemy_planets:
        seeded = world.best_probe_aim(
            enemy.id,
            planet.id,
            max(1, int(enemy.ships)),
        )
        if seeded is None:
            continue
        _, aim = seeded
        eta = aim[1]
        if eta > MULTI_ENEMY_PROACTIVE_HORIZON:
            continue
        threats.append((eta, int(enemy.ships)))

    if not threats:
        return 0

    threats.sort()
    best_stacked = 0
    left = 0
    running = 0
    for right in range(len(threats)):
        running += threats[right][1]
        while threats[right][0] - threats[left][0] > MULTI_ENEMY_STACK_WINDOW:
            running -= threats[left][1]
            left += 1
        best_stacked = max(best_stacked, running)

    return int(best_stacked * MULTI_ENEMY_PROACTIVE_RATIO)


def swarm_eta_tolerance(options, target, world):
    if len(options) >= 3:
        return THREE_SOURCE_ETA_TOLERANCE
    if target.owner not in (-1, world.player):
        return HOSTILE_SWARM_ETA_TOLERANCE
    return MULTI_SOURCE_ETA_TOLERANCE


def detect_enemy_crashes(world):
    crashes = []
    for target_id, arrivals in world.arrivals_by_planet.items():
        enemy_events = [
            (int(math.ceil(eta)), owner, int(ships))
            for eta, owner, ships in arrivals
            if owner not in (-1, world.player) and ships > 0
        ]
        enemy_events.sort()
        for i in range(len(enemy_events)):
            eta_a, owner_a, ships_a = enemy_events[i]
            for j in range(i + 1, len(enemy_events)):
                eta_b, owner_b, ships_b = enemy_events[j]
                if owner_a == owner_b:
                    continue
                if abs(eta_a - eta_b) > CRASH_EXPLOIT_ETA_WINDOW:
                    break
                if ships_a + ships_b < CRASH_EXPLOIT_MIN_TOTAL_SHIPS:
                    continue
                crashes.append(
                    {
                        "target_id": target_id,
                        "crash_turn": max(eta_a, eta_b),
                        "owners": (owner_a, owner_b),
                        "ships": (ships_a, ships_b),
                    }
                )
    return crashes


def build_policy_state(world, deadline=None):
    def expired():
        return deadline is not None and time.perf_counter() > deadline

    indirect_wealth_map = {}
    for target_id, features in world.indirect_feature_map.items():
        friendly, neutral, enemy = features
        indirect_wealth_map[target_id] = (
            friendly * INDIRECT_FRIENDLY_WEIGHT
            + neutral * INDIRECT_NEUTRAL_WEIGHT
            + enemy * INDIRECT_ENEMY_WEIGHT
        )

    reserve = {}
    attack_budget = {}
    reaction_time_map = {}

    for target in world.planets:
        if expired():
            break
        if target.owner == world.player:
            continue
        my_sources = nearest_sources_to_target(target, world.my_planets, REACTION_SOURCE_TOP_K_MY)
        enemy_sources = nearest_sources_to_target(target, world.enemy_planets, REACTION_SOURCE_TOP_K_ENEMY)
        my_t = min_legal_reaction_time(target, my_sources, world)
        enemy_t = min_legal_reaction_time(target, enemy_sources, world)
        reaction_time_map[target.id] = (my_t, enemy_t)

    for planet in world.my_planets:
        if expired():
            break
        exact_keep = world.keep_needed_map.get(planet.id, 0)

        proactive_keep = 0
        for enemy in nearest_sources_to_target(planet, world.enemy_planets, PROACTIVE_ENEMY_TOP_K):
            enemy_aim = world.plan_shot(enemy.id, planet.id, max(1, int(enemy.ships)))
            if enemy_aim is None:
                continue
            enemy_eta = enemy_aim[1]
            if enemy_eta > PROACTIVE_DEFENSE_HORIZON:
                continue
            proactive_keep = max(
                proactive_keep,
                int(enemy.ships * PROACTIVE_DEFENSE_RATIO),
            )
        proactive_keep = max(proactive_keep, stacked_enemy_proactive_keep(planet, world))

        reserve[planet.id] = min(int(planet.ships), max(exact_keep, proactive_keep))
        attack_budget[planet.id] = max(0, int(planet.ships) - reserve[planet.id])

    return {
        "indirect_wealth_map": indirect_wealth_map,
        "reserve": reserve,
        "attack_budget": attack_budget,
        "reaction_time_map": reaction_time_map,
    }


def build_modes(world):
    domination = (world.my_total - world.enemy_total) / max(1, world.my_total + world.enemy_total)
    is_behind = domination < BEHIND_DOMINATION
    is_ahead = domination > AHEAD_DOMINATION
    is_dominating = is_ahead or (
        world.max_enemy_strength > 0 and world.my_total > world.max_enemy_strength * 1.25
    )
    is_finishing = (
        domination > FINISHING_DOMINATION
        and world.my_prod > world.enemy_prod * FINISHING_PROD_RATIO
        and world.step > 100
    )

    attack_margin_mult = 1.0
    if is_ahead:
        attack_margin_mult += AHEAD_ATTACK_MARGIN_BONUS
    if is_behind:
        attack_margin_mult -= BEHIND_ATTACK_MARGIN_PENALTY
    if is_finishing:
        attack_margin_mult += FINISHING_ATTACK_MARGIN_BONUS

    return {
        "domination": domination,
        "is_behind": is_behind,
        "is_ahead": is_ahead,
        "is_dominating": is_dominating,
        "is_finishing": is_finishing,
        "attack_margin_mult": attack_margin_mult,
    }


def is_safe_neutral(target, policy):
    if target.owner != -1:
        return False
    my_t, enemy_t = policy_reaction_times(target.id, policy)
    return my_t <= enemy_t - SAFE_NEUTRAL_MARGIN


def is_contested_neutral(target, policy):
    if target.owner != -1:
        return False
    my_t, enemy_t = policy_reaction_times(target.id, policy)
    return abs(my_t - enemy_t) <= CONTESTED_NEUTRAL_MARGIN


def opening_filter(target, arrival_turns, needed, src_available, world, policy):
    if not world.is_opening or target.owner != -1:
        return False
    if target.id in world.comet_ids:
        return False
    if world.is_static(target.id):
        return False

    my_t, enemy_t = policy_reaction_times(target.id, policy)
    reaction_gap = enemy_t - my_t
    if (
        target.production >= SAFE_OPENING_PROD_THRESHOLD
        and arrival_turns <= SAFE_OPENING_TURN_LIMIT
        and reaction_gap >= SAFE_NEUTRAL_MARGIN
    ):
        return False

    if world.is_four_player:
        affordable = needed <= max(
            PARTIAL_SOURCE_MIN_SHIPS,
            int(src_available * FOUR_PLAYER_ROTATING_SEND_RATIO),
        )
        if (
            affordable
            and arrival_turns <= FOUR_PLAYER_ROTATING_TURN_LIMIT
            and reaction_gap >= FOUR_PLAYER_ROTATING_REACTION_GAP
        ):
            return False
        return True

    return arrival_turns > ROTATING_OPENING_MAX_TURNS or target.production <= ROTATING_OPENING_LOW_PROD


def target_value(target, arrival_turns, mission, world, modes, policy):
    turns_profit = max(1, world.remaining_steps - arrival_turns)
    if target.id in world.comet_ids:
        life = world.comet_life(target.id)
        turns_profit = max(0, min(turns_profit, life - arrival_turns))
        if turns_profit <= 0:
            return -1.0

    value = target.production * turns_profit
    value += policy["indirect_wealth_map"][target.id] * turns_profit * INDIRECT_VALUE_SCALE

    if world.is_static(target.id):
        value *= STATIC_NEUTRAL_VALUE_MULT if target.owner == -1 else STATIC_HOSTILE_VALUE_MULT
    else:
        value *= ROTATING_OPENING_VALUE_MULT if world.is_opening else 1.0

    if target.owner not in (-1, world.player):
        value *= OPENING_HOSTILE_TARGET_VALUE_MULT if world.is_opening else HOSTILE_TARGET_VALUE_MULT

    if target.owner == -1:
        if is_safe_neutral(target, policy):
            value *= SAFE_NEUTRAL_VALUE_MULT
        elif is_contested_neutral(target, policy):
            value *= CONTESTED_NEUTRAL_VALUE_MULT
        if world.is_early:
            value *= EARLY_NEUTRAL_VALUE_MULT

    if target.id in world.comet_ids:
        value *= COMET_VALUE_MULT

    if mission == "snipe":
        value *= SNIPE_VALUE_MULT
    elif mission == "swarm":
        value *= SWARM_VALUE_MULT
    elif mission == "reinforce":
        value *= REINFORCE_VALUE_MULT
    elif mission == "crash_exploit":
        value *= CRASH_EXPLOIT_VALUE_MULT

    if world.is_late:
        value += max(0, target.ships) * LATE_IMMEDIATE_SHIP_VALUE

    # Elimination targeting: preferentially attack weak enemies to eliminate
    # them, regardless of game phase (extended from late-only).
    if target.owner not in (-1, world.player):
        enemy_strength = world.owner_strength.get(target.owner, 0)
        if enemy_strength <= WEAK_ENEMY_THRESHOLD:
            value += ELIMINATION_BONUS

    if modes["is_finishing"] and target.owner not in (-1, world.player):
        value *= FINISHING_HOSTILE_VALUE_MULT
    if modes["is_behind"] and target.owner == -1 and not world.is_static(target.id):
        value *= BEHIND_ROTATING_NEUTRAL_VALUE_MULT
    if modes["is_behind"] and target.owner == -1 and is_safe_neutral(target, policy):
        value *= 1.08
    if modes["is_dominating"] and target.owner == -1 and is_contested_neutral(target, policy):
        value *= 0.92

    return value


def reinforce_value(target, hold_until, world, policy):
    saved_turns = max(1, world.remaining_steps - hold_until)
    value = target.production * saved_turns + max(0, target.ships) * DEFENSE_SHIP_VALUE
    if world.enemy_planets and nearest_distance_to_set(target.x, target.y, world.enemy_planets) < 22:
        value *= DEFENSE_FRONTIER_SCORE_MULT
    value += policy["indirect_wealth_map"][target.id] * saved_turns * INDIRECT_VALUE_SCALE * 0.35
    return value * REINFORCE_VALUE_MULT


def preferred_send(target, base_needed, arrival_turns, src_available, world, modes, policy):
    send = max(base_needed, int(math.ceil(base_needed * modes["attack_margin_mult"])))
    margin = 0
    if target.owner == -1:
        margin += min(
            NEUTRAL_MARGIN_CAP,
            NEUTRAL_MARGIN_BASE + target.production * NEUTRAL_MARGIN_PROD_WEIGHT,
        )
    else:
        margin += min(
            HOSTILE_MARGIN_CAP,
            HOSTILE_MARGIN_BASE + target.production * HOSTILE_MARGIN_PROD_WEIGHT,
        )
    if world.is_static(target.id):
        margin += STATIC_TARGET_MARGIN
    if is_contested_neutral(target, policy):
        margin += CONTESTED_TARGET_MARGIN
    if world.is_four_player:
        margin += FOUR_PLAYER_TARGET_MARGIN
    if arrival_turns > LONG_TRAVEL_MARGIN_START:
        margin += min(LONG_TRAVEL_MARGIN_CAP, arrival_turns // LONG_TRAVEL_MARGIN_DIVISOR)
    if target.id in world.comet_ids:
        margin = max(0, margin - COMET_MARGIN_RELIEF)
    if modes["is_finishing"] and target.owner not in (-1, world.player):
        margin += FINISHING_HOSTILE_SEND_BONUS
    return min(src_available, send + margin)


def apply_score_modifiers(base_score, target, mission, world):
    score = base_score
    if world.is_static(target.id):
        score *= STATIC_TARGET_SCORE_MULT
    if world.is_early and target.owner == -1 and world.is_static(target.id):
        score *= EARLY_STATIC_NEUTRAL_SCORE_MULT
    if world.is_four_player and target.owner == -1 and not world.is_static(target.id):
        score *= FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT
    if (
        len(world.static_neutral_planets) >= DENSE_STATIC_NEUTRAL_COUNT
        and target.owner == -1
        and not world.is_static(target.id)
    ):
        score *= DENSE_ROTATING_NEUTRAL_SCORE_MULT
    if mission == "snipe":
        score *= SNIPE_SCORE_MULT
    elif mission == "swarm":
        score *= SWARM_SCORE_MULT
    elif mission == "crash_exploit":
        score *= CRASH_EXPLOIT_SCORE_MULT
    return score


def settle_plan(
    src,
    target,
    src_cap,
    send_guess,
    world,
    planned_commitments,
    modes,
    policy,
    mission="capture",
    eval_turn_fn=None,
    anchor_turn=None,
    anchor_tolerance=None,
    max_iter=4,
):
    if src_cap < 1:
        return None

    seed_hint = max(1, min(src_cap, int(send_guess)))
    eval_turn_fn = eval_turn_fn or (lambda turns: turns)
    anchor_tolerance = (
        anchor_tolerance
        if anchor_tolerance is not None
        else (1 if mission == "snipe" else None)
    )
    tested = {}
    tested_order = []

    def evaluate(send):
        send = max(1, min(src_cap, int(send)))
        cached = tested.get(send)
        if cached is not None or send in tested:
            return cached

        aim = world.plan_shot(src.id, target.id, send)
        if aim is None:
            tested[send] = None
            return None

        angle, turns, _, _ = aim
        if mission == "crash_exploit" and anchor_turn is not None and turns < anchor_turn:
            tested[send] = None
            return None
        raw_eval_turn = int(math.ceil(eval_turn_fn(turns)))
        if raw_eval_turn < turns:
            tested[send] = None
            return None
        eval_turn = raw_eval_turn
        need = world.min_ships_to_own_by(
            target.id,
            eval_turn,
            world.player,
            arrival_turn=turns,
            planned_commitments=planned_commitments,
            upper_bound=src_cap,
        )
        if need <= 0 or need > src_cap:
            tested[send] = None
            return None

        if mission in ("snipe", "crash_exploit"):
            desired = need
        elif mission == "rescue":
            desired = min(
                src_cap,
                max(
                    need,
                    need + DEFENSE_SEND_MARGIN_BASE + target.production * DEFENSE_SEND_MARGIN_PROD_WEIGHT,
                ),
            )
        else:
            desired = min(
                src_cap,
                max(need, preferred_send(target, need, turns, src_cap, world, modes, policy)),
            )

        result = (angle, turns, eval_turn, need, send, desired)
        tested[send] = result
        tested_order.append(send)
        return result

    initial_candidates = sorted(
        world.probe_ship_candidates(
            src.id,
            target.id,
            src_cap,
            hints=(seed_hint,),
        ),
        key=lambda send: (abs(send - seed_hint), send),
    )

    current_send = None
    for seed in initial_candidates:
        result = evaluate(seed)
        if result is None:
            continue
        if (
            anchor_turn is not None
            and anchor_tolerance is not None
            and abs(result[1] - anchor_turn) > anchor_tolerance
        ):
            continue
        current_send = seed
        break

    if current_send is None:
        return None

    for _ in range(max_iter):
        result = evaluate(current_send)
        if result is None:
            break

        angle, turns, eval_turn, need, actual_send, desired = result
        if desired == actual_send:
            if (
                anchor_turn is not None
                and anchor_tolerance is not None
                and abs(turns - anchor_turn) > anchor_tolerance
            ):
                return None
            if mission == "rescue" and turns > eval_turn:
                return None
            return angle, turns, eval_turn, need, actual_send

        next_send = max(1, min(src_cap, int(desired)))
        if next_send in tested:
            current_send = next_send
            break
        current_send = next_send

    candidate_sends = sorted(
        [send for send in tested_order if tested.get(send) is not None],
        key=lambda send: (
            0
            if mission != "snipe" or anchor_turn is None
            else abs(tested[send][1] - anchor_turn),
            abs(send - seed_hint),
            tested[send][1],
            send,
        ),
    )

    seen = set()
    for send in candidate_sends:
        if send in seen:
            continue
        seen.add(send)
        result = tested.get(send)
        if result is None:
            continue
        angle, turns, eval_turn, need, actual_send, _ = result
        if actual_send < need:
            continue
        if (
            anchor_turn is not None
            and anchor_tolerance is not None
            and abs(turns - anchor_turn) > anchor_tolerance
        ):
            continue
        if mission == "rescue" and turns > eval_turn:
            continue
        return angle, turns, eval_turn, need, actual_send

    return None


def settle_reinforce_plan(
    src,
    target,
    src_cap,
    send_guess,
    world,
    planned_commitments,
    hold_until,
    max_arrival_turn,
    max_iter=4,
):
    if src_cap < 1:
        return None

    seed_hint = max(1, min(src_cap, int(send_guess)))
    tested = {}
    tested_order = []

    def evaluate(send):
        send = max(1, min(src_cap, int(send)))
        cached = tested.get(send)
        if cached is not None or send in tested:
            return cached

        aim = world.plan_shot(src.id, target.id, send)
        if aim is None:
            tested[send] = None
            return None

        angle, turns, _, _ = aim
        if turns > max_arrival_turn:
            tested[send] = None
            return None

        need = world.reinforcement_needed_to_hold_until(
            target.id,
            turns,
            hold_until,
            planned_commitments=planned_commitments,
            upper_bound=src_cap,
        )
        if need <= 0 or need > src_cap:
            tested[send] = None
            return None

        desired = min(src_cap, need + REINFORCE_SAFETY_MARGIN)
        result = (angle, turns, hold_until, need, send, desired)
        tested[send] = result
        tested_order.append(send)
        return result

    initial_candidates = sorted(
        world.probe_ship_candidates(
            src.id,
            target.id,
            src_cap,
            hints=(seed_hint,),
        ),
        key=lambda send: (abs(send - seed_hint), send),
    )

    current_send = None
    for seed in initial_candidates:
        result = evaluate(seed)
        if result is None:
            continue
        current_send = seed
        break

    if current_send is None:
        return None

    for _ in range(max_iter):
        result = evaluate(current_send)
        if result is None:
            break

        angle, turns, eval_turn, need, actual_send, desired = result
        if desired == actual_send:
            return angle, turns, eval_turn, need, actual_send

        next_send = max(1, min(src_cap, int(desired)))
        if next_send in tested:
            current_send = next_send
            break
        current_send = next_send

    candidate_sends = sorted(
        [send for send in tested_order if tested.get(send) is not None],
        key=lambda send: (abs(send - seed_hint), tested[send][1], send),
    )
    for send in candidate_sends:
        result = tested.get(send)
        if result is None:
            continue
        angle, turns, eval_turn, need, actual_send, _ = result
        if actual_send < need or turns > max_arrival_turn:
            continue
        return angle, turns, eval_turn, need, actual_send

    return None


def build_snipe_mission(src, target, src_available, world, planned_commitments, modes, policy):
    if target.owner != -1:
        return None

    enemy_etas = sorted(
        {
            int(math.ceil(eta))
            for eta, owner, ships in world.arrivals_by_planet.get(target.id, [])
            if owner not in (-1, world.player) and ships > 0
        }
    )
    if not enemy_etas:
        return None

    best = None
    for enemy_eta in enemy_etas[:3]:
        seeded = world.best_probe_aim(
            src.id,
            target.id,
            src_available,
            hints=(int(target.ships) + 1, int(target.ships) + 8),
            anchor_turn=enemy_eta,
            max_anchor_diff=1,
        )
        if seeded is None:
            continue

        probe, rough = seeded
        sync_turn = max(rough[1], enemy_eta)
        if target.id in world.comet_ids:
            life = world.comet_life(target.id)
            if sync_turn >= life or sync_turn > COMET_MAX_CHASE_TURNS:
                continue

        plan = settle_plan(
            src,
            target,
            src_available,
            probe,
            world,
            planned_commitments,
            modes,
            policy,
            mission="snipe",
            eval_turn_fn=lambda turns, enemy_eta=enemy_eta: max(turns, enemy_eta),
            anchor_turn=enemy_eta,
        )
        if plan is None:
            continue

        angle, turns, sync_turn, need, send_pref = plan
        if target.id in world.comet_ids:
            life = world.comet_life(target.id)
            if sync_turn >= life or sync_turn > COMET_MAX_CHASE_TURNS:
                continue

        value = target_value(target, sync_turn, "snipe", world, modes, policy)
        if value <= 0:
            continue

        score = apply_score_modifiers(
            value / (send_pref + sync_turn * SNIPE_COST_TURN_WEIGHT + 1.0),
            target,
            "snipe",
            world,
        )
        option = ShotOption(
            score=score,
            src_id=src.id,
            target_id=target.id,
            angle=angle,
            turns=turns,
            needed=need,
            send_cap=send_pref,
            mission="snipe",
            anchor_turn=enemy_eta,
        )
        mission_obj = Mission(
            kind="snipe",
            score=score,
            target_id=target.id,
            turns=sync_turn,
            options=[option],
        )
        if best is None or mission_obj.score > best.score:
            best = mission_obj

    return best


def build_rescue_missions(world, policy, planned_commitments, modes):
    missions = []

    for target in world.my_planets:
        fall_turn = world.fall_turn_map.get(target.id)
        if fall_turn is None or fall_turn > DEFENSE_LOOKAHEAD_TURNS:
            continue

        for src in world.my_planets:
            if src.id == target.id:
                continue

            src_available = policy["attack_budget"].get(src.id, 0)
            if src_available < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(target.production + DEFENSE_SEND_MARGIN_BASE + 2,),
                max_turn=fall_turn,
            )
            if seeded is None:
                continue
            probe, probe_aim = seeded
            plan = settle_plan(
                src,
                target,
                src_available,
                probe,
                world,
                planned_commitments,
                modes,
                policy,
                mission="rescue",
                eval_turn_fn=lambda _turns, fall_turn=fall_turn: fall_turn,
                anchor_turn=fall_turn,
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            saved_turns = max(1, world.remaining_steps - fall_turn)
            value = target.production * saved_turns + max(0, target.ships) * DEFENSE_SHIP_VALUE
            if world.enemy_planets and nearest_distance_to_set(target.x, target.y, world.enemy_planets) < 22:
                value *= DEFENSE_FRONTIER_SCORE_MULT
            score = value / (send_pref + turns * DEFENSE_COST_TURN_WEIGHT + 1.0)

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="rescue",
                anchor_turn=fall_turn,
            )
            missions.append(
                Mission(
                    kind="rescue",
                    score=score,
                    target_id=target.id,
                    turns=fall_turn,
                    options=[option],
                )
            )

    return missions


def build_recapture_missions(world, policy, planned_commitments, modes):
    missions = []

    for target in world.my_planets:
        fall_turn = world.fall_turn_map.get(target.id)
        if fall_turn is None or fall_turn > DEFENSE_LOOKAHEAD_TURNS:
            continue

        for src in world.my_planets:
            if src.id == target.id:
                continue

            src_available = policy["attack_budget"].get(src.id, 0)
            if src_available < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(target.production + DEFENSE_SEND_MARGIN_BASE + 2,),
                min_turn=fall_turn + 1,
                max_turn=fall_turn + RECAPTURE_LOOKAHEAD_TURNS,
            )
            if seeded is None:
                continue
            probe, probe_aim = seeded
            probe_turns = probe_aim[1]

            plan = settle_plan(
                src,
                target,
                src_available,
                probe,
                world,
                planned_commitments,
                modes,
                policy,
                mission="capture",
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            if turns <= fall_turn or turns - fall_turn > RECAPTURE_LOOKAHEAD_TURNS:
                continue

            saved_turns = max(1, world.remaining_steps - turns)
            value = (
                RECAPTURE_PRODUCTION_WEIGHT * target.production * saved_turns
                + RECAPTURE_IMMEDIATE_WEIGHT * max(0, target.ships)
            )
            if world.enemy_planets and nearest_distance_to_set(target.x, target.y, world.enemy_planets) < 22:
                value *= RECAPTURE_FRONTIER_MULT
            value *= RECAPTURE_VALUE_MULT
            score = value / (send_pref + turns * RECAPTURE_COST_TURN_WEIGHT + 1.0)

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="recapture",
                anchor_turn=fall_turn,
            )
            missions.append(
                Mission(
                    kind="recapture",
                    score=score,
                    target_id=target.id,
                    turns=turns,
                    options=[option],
                )
            )

    return missions


def build_reinforce_missions(world, policy, planned_commitments, modes, inventory_left_fn):
    if not REINFORCE_ENABLED:
        return []

    missions = []
    if world.remaining_steps < REINFORCE_MIN_FUTURE_TURNS:
        return missions

    for target in world.my_planets:
        fall_turn = world.fall_turn_map.get(target.id)
        if fall_turn is None:
            continue
        if target.production < REINFORCE_MIN_PRODUCTION:
            continue

        hold_until = min(HORIZON, fall_turn + REINFORCE_HOLD_LOOKAHEAD)
        max_arrival_turn = min(fall_turn, REINFORCE_MAX_TRAVEL_TURNS)

        for src in world.my_planets:
            if src.id == target.id:
                continue

            budget = inventory_left_fn(src.id)
            source_cap = min(budget, int(src.ships * REINFORCE_MAX_SOURCE_FRACTION))
            if source_cap < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                source_cap,
                hints=(target.production + REINFORCE_SAFETY_MARGIN + 2,),
                max_turn=max_arrival_turn,
            )
            if seeded is None:
                continue
            probe, _ = seeded

            plan = settle_reinforce_plan(
                src,
                target,
                source_cap,
                probe,
                world,
                planned_commitments,
                hold_until,
                max_arrival_turn,
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            value = reinforce_value(target, hold_until, world, policy)
            score = value / (send_pref + turns * REINFORCE_COST_TURN_WEIGHT + 1.0)

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="reinforce",
                anchor_turn=hold_until,
            )
            missions.append(
                Mission(
                    kind="reinforce",
                    score=score,
                    target_id=target.id,
                    turns=fall_turn,
                    options=[option],
                )
            )

    return missions


def build_crash_exploit_missions(world, policy, planned_commitments, modes):
    if not CRASH_EXPLOIT_ENABLED or not world.is_four_player:
        return []

    missions = []
    for crash in detect_enemy_crashes(world):
        target = world.planet_by_id[crash["target_id"]]
        if target.owner == world.player:
            continue
        desired_arrival = crash["crash_turn"] + CRASH_EXPLOIT_POST_CRASH_DELAY

        for src in world.my_planets:
            src_available = policy["attack_budget"].get(src.id, 0)
            if src_available < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(12, int(target.ships) + 1),
                anchor_turn=desired_arrival,
                max_anchor_diff=CRASH_EXPLOIT_ETA_WINDOW,
            )
            if seeded is None:
                continue
            probe, _ = seeded

            plan = settle_plan(
                src,
                target,
                src_available,
                probe,
                world,
                planned_commitments,
                modes,
                policy,
                mission="crash_exploit",
                eval_turn_fn=lambda turns, desired_arrival=desired_arrival: max(turns, desired_arrival),
                anchor_turn=desired_arrival,
                anchor_tolerance=CRASH_EXPLOIT_ETA_WINDOW,
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
                continue
            value = target_value(target, turns, "crash_exploit", world, modes, policy)
            if value <= 0:
                continue

            score = apply_score_modifiers(
                value / (send_pref + turns * SNIPE_COST_TURN_WEIGHT + 1.0),
                target,
                "crash_exploit",
                world,
            )
            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="crash_exploit",
                anchor_turn=desired_arrival,
            )
            missions.append(
                Mission(
                    kind="crash_exploit",
                    score=score,
                    target_id=target.id,
                    turns=turns,
                    options=[option],
                )
            )

    return missions

def plan_moves(world, deadline=None):
    def expired():
        return deadline is not None and time.perf_counter() > deadline

    def time_left():
        if deadline is None:
            return 10**9
        return deadline - time.perf_counter()

    def allow_heavy_phase():
        return time_left() > HEAVY_PHASE_MIN_TIME and len(world.planets) <= HEAVY_ROUTE_PLANET_LIMIT

    def allow_optional_phase():
        return time_left() > OPTIONAL_PHASE_MIN_TIME

    modes = build_modes(world)
    policy = build_policy_state(world, deadline=deadline)
    planned_commitments = defaultdict(list)
    source_options_by_target = defaultdict(list)
    missions = []
    moves = []
    spent_total = defaultdict(int)

    def source_inventory_left(source_id):
        return world.source_inventory_left(source_id, spent_total)

    def source_attack_left(source_id):
        budget = policy["attack_budget"].get(source_id, 0)
        return max(0, budget - spent_total[source_id])

    def append_move(src_id, angle, ships):
        send = min(int(ships), source_inventory_left(src_id))
        if send < 1:
            return 0
        moves.append([src_id, float(angle), int(send)])
        spent_total[src_id] += send
        return send

    def finalize_moves():
        final_moves = []
        used_final = defaultdict(int)
        for src_id, angle, ships in moves:
            source = world.planet_by_id[src_id]
            max_allowed = int(source.ships) - used_final[src_id]
            send = min(int(ships), max_allowed)
            if send >= 1:
                final_moves.append([src_id, float(angle), int(send)])
                used_final[src_id] += send
        return final_moves

    def compute_live_doomed():
        doomed = set()
        for planet in world.my_planets:
            status = world.hold_status(
                planet.id,
                planned_commitments=planned_commitments,
                horizon=DOOMED_EVAC_TURN_LIMIT,
            )
            if (
                not status["holds_full"]
                and status["fall_turn"] is not None
                and status["fall_turn"] <= DOOMED_EVAC_TURN_LIMIT
                and source_inventory_left(planet.id) >= DOOMED_MIN_SHIPS
            ):
                doomed.add(planet.id)
        return doomed

    def time_filters_pass(target, turns, needed, src_cap):
        if not candidate_time_valid(target, turns, world, VERY_LATE_CAPTURE_BUFFER if world.is_very_late else LATE_CAPTURE_BUFFER):
            return False
        if opening_filter(target, turns, needed, src_cap, world, policy):
            return False
        return True

    # ---------------------------------------------------------------
    # Quick reinforcement pass: shore up underdefended planets by
    # transferring from nearby surplus planets before missions are built.
    # (Hyperion-style: weak < 12 ships, strong > 25 ships, range ≤ 25)
    # ---------------------------------------------------------------
    if allow_optional_phase():
        underdefended = []
        surplus_planets = []
        for planet in world.my_planets:
            keep = policy["reserve"].get(planet.id, 0)
            budget = policy["attack_budget"].get(planet.id, 0)
            available = int(planet.ships)
            if keep > available:
                # Planet needs more ships than it has
                underdefended.append((planet, keep - available))
            elif budget > 0 and available > 25:
                # Planet has surplus attack budget
                surplus_planets.append((planet, budget))

        for weak_planet, deficit in sorted(underdefended, key=lambda x: -x[1]):
            if expired():
                return finalize_moves()
            best_source = None
            best_dist = 10**9
            for strong_planet, strong_budget in surplus_planets:
                d = planet_distance(weak_planet, strong_planet)
                if d > 25:
                    continue
                if source_attack_left(strong_planet.id) <= 0:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_source = (strong_planet, strong_budget)
            if best_source is None:
                continue
            strong_planet, strong_budget = best_source
            transfer = min(
                int(deficit),
                source_attack_left(strong_planet.id),
                int(strong_budget * 0.6),
            )
            if transfer < 1:
                continue
            aim = world.plan_shot(strong_planet.id, weak_planet.id, transfer)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            actual = append_move(strong_planet.id, angle, transfer)
            if actual >= 1:
                planned_commitments[weak_planet.id].append((turns, world.player, int(actual)))

    if allow_heavy_phase():
        missions.extend(
            build_reinforce_missions(
                world,
                policy,
                planned_commitments,
                modes,
                source_inventory_left,
            )
        )
    missions.extend(build_rescue_missions(world, policy, planned_commitments, modes))
    missions.extend(build_recapture_missions(world, policy, planned_commitments, modes))

    # Only build candidates after solving an intercept so timing decisions come
    # from a real route.
    for src in world.my_planets:
        if expired():
            return finalize_moves()
        src_available = source_attack_left(src.id)
        if src_available <= 0:
            continue

        for target in world.planets:
            if expired():
                return finalize_moves()
            if target.id == src.id or target.owner == world.player:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(int(target.ships) + 1,),
            )
            if seeded is None:
                continue
            _, rough_aim = seeded

            rough_turns = rough_aim[1]
            if not candidate_time_valid(
                target,
                rough_turns,
                world,
                VERY_LATE_CAPTURE_BUFFER if world.is_very_late else LATE_CAPTURE_BUFFER,
            ):
                continue

            global_needed = world.min_ships_to_own_at(
                target.id,
                rough_turns,
                world.player,
                planned_commitments=planned_commitments,
            )
            if global_needed <= 0:
                continue
            if opening_filter(target, rough_turns, global_needed, src_available, world, policy):
                continue

            partial_send_cap = min(
                src_available,
                preferred_send(
                    target,
                    global_needed,
                    rough_turns,
                    src_available,
                    world,
                    modes,
                    policy,
                ),
            )
            if partial_send_cap >= PARTIAL_SOURCE_MIN_SHIPS:
                partial_seed = world.best_probe_aim(
                    src.id,
                    target.id,
                    partial_send_cap,
                    hints=(partial_send_cap, global_needed, int(target.ships) + 1),
                )
                if partial_seed is not None:
                    _, partial_aim = partial_seed
                    p_angle, p_turns, _, _ = partial_aim
                    if time_filters_pass(target, p_turns, global_needed, src_available):
                        partial_value = target_value(target, p_turns, "swarm", world, modes, policy)
                        if partial_value > 0:
                            partial_score = apply_score_modifiers(
                                partial_value / (partial_send_cap + p_turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                                target,
                                "swarm",
                                world,
                            )
                            source_options_by_target[target.id].append(
                                ShotOption(
                                    score=partial_score,
                                    src_id=src.id,
                                    target_id=target.id,
                                    angle=p_angle,
                                    turns=p_turns,
                                    needed=global_needed,
                                    send_cap=partial_send_cap,
                                    mission="swarm",
                                )
                            )

            if global_needed <= src_available:
                send_guess = preferred_send(
                    target,
                    global_needed,
                    rough_turns,
                    src_available,
                    world,
                    modes,
                    policy,
                )
                plan = settle_plan(
                    src,
                    target,
                    src_available,
                    send_guess,
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                )
                if plan is None:
                    continue

                angle, turns, _, needed, send_cap = plan
                if not time_filters_pass(target, turns, needed, src_available):
                    continue
                if send_cap < 1:
                    continue

                value = target_value(target, turns, "capture", world, modes, policy)
                if value <= 0:
                    continue

                score = apply_score_modifiers(
                    value / (send_cap + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "capture",
                    world,
                )

                option = ShotOption(
                    score=score,
                    src_id=src.id,
                    target_id=target.id,
                    angle=angle,
                    turns=turns,
                    needed=needed,
                    send_cap=send_cap,
                    mission="capture",
                )

                if send_cap >= needed:
                    missions.append(
                        Mission(
                            kind="single",
                            score=score,
                            target_id=target.id,
                            turns=turns,
                            options=[option],
                        )
                    )

            snipe = build_snipe_mission(src, target, src_available, world, planned_commitments, modes, policy)
            if snipe is not None:
                missions.append(snipe)

    # Allow small synchronized two-source finishes when one source is not
    # enough on its own.
    for target_id, options in source_options_by_target.items():
        if expired():
            return finalize_moves()
        if len(options) < 2:
            continue

        target = world.planet_by_id[target_id]
        top_options = sorted(options, key=lambda item: -item.score)[:MULTI_SOURCE_TOP_K]
        for i in range(len(top_options)):
            for j in range(i + 1, len(top_options)):
                first = top_options[i]
                second = top_options[j]
                if first.src_id == second.src_id:
                    continue
                pair_tol = swarm_eta_tolerance((first, second), target, world)
                if abs(first.turns - second.turns) > pair_tol:
                    continue

                joint_turn = max(first.turns, second.turns)
                total_cap = first.send_cap + second.send_cap
                need = world.min_ships_to_own_at(
                    target_id,
                    joint_turn,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=total_cap,
                )
                if need <= 0:
                    continue
                if first.send_cap >= need or second.send_cap >= need:
                    continue
                if total_cap < need:
                    continue

                value = target_value(target, joint_turn, "swarm", world, modes, policy)
                if value <= 0:
                    continue

                pair_score = apply_score_modifiers(
                    value / (need + joint_turn * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "swarm",
                    world,
                )
                pair_score *= MULTI_SOURCE_PLAN_PENALTY
                missions.append(
                    Mission(
                        kind="swarm",
                        score=pair_score,
                        target_id=target_id,
                        turns=joint_turn,
                        options=[first, second],
                    )
                )

        if (
            THREE_SOURCE_SWARM_ENABLED
            and allow_heavy_phase()
            and target.owner not in (-1, world.player)
            and int(target.ships) >= THREE_SOURCE_MIN_TARGET_SHIPS
            and len(top_options) >= 3
        ):
            for i in range(len(top_options)):
                for j in range(i + 1, len(top_options)):
                    for k in range(j + 1, len(top_options)):
                        if expired():
                            return finalize_moves()
                        trio = [top_options[i], top_options[j], top_options[k]]
                        if len({option.src_id for option in trio}) < 3:
                            continue
                        trio_tol = swarm_eta_tolerance(tuple(trio), target, world)
                        turns = [option.turns for option in trio]
                        if max(turns) - min(turns) > trio_tol:
                            continue

                        joint_turn = max(turns)
                        total_cap = sum(option.send_cap for option in trio)
                        need = world.min_ships_to_own_at(
                            target_id,
                            joint_turn,
                            world.player,
                            planned_commitments=planned_commitments,
                            upper_bound=total_cap,
                        )
                        if need <= 0 or total_cap < need:
                            continue
                        if any(
                            trio[a].send_cap + trio[b].send_cap >= need
                            for a in range(3)
                            for b in range(a + 1, 3)
                        ):
                            continue

                        value = target_value(target, joint_turn, "swarm", world, modes, policy)
                        if value <= 0:
                            continue

                        trio_score = apply_score_modifiers(
                            value / (need + joint_turn * ATTACK_COST_TURN_WEIGHT + 1.0),
                            target,
                            "swarm",
                            world,
                        )
                        trio_score *= THREE_SOURCE_PLAN_PENALTY
                        missions.append(
                            Mission(
                                kind="swarm",
                                score=trio_score,
                                target_id=target_id,
                                turns=joint_turn,
                                options=trio,
                            )
                        )

    if allow_heavy_phase():
        missions.extend(build_crash_exploit_missions(world, policy, planned_commitments, modes))

    missions.sort(key=lambda item: -item.score)

    # Update commitments after every accepted launch so later plans see the
    # timing that is already spoken for.
    for mission in missions:
        if expired():
            return finalize_moves()
        target = world.planet_by_id[mission.target_id]

        if mission.kind in ("single", "snipe", "rescue", "recapture", "reinforce", "crash_exploit"):
            option = mission.options[0]
            src = world.planet_by_id[option.src_id]
            if mission.kind == "reinforce":
                left = min(
                    source_inventory_left(option.src_id),
                    int(src.ships * REINFORCE_MAX_SOURCE_FRACTION),
                )
            else:
                left = source_attack_left(option.src_id)
            if left <= 0:
                continue

            if mission.kind == "reinforce":
                plan = settle_reinforce_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    option.anchor_turn,
                    mission.turns,
                )
            elif mission.kind == "rescue":
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="rescue",
                    eval_turn_fn=lambda _turns, hold_turn=mission.turns: hold_turn,
                    anchor_turn=option.anchor_turn,
                )
            elif mission.kind == "snipe":
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="snipe",
                    eval_turn_fn=lambda turns, enemy_eta=option.anchor_turn: max(turns, enemy_eta),
                    anchor_turn=option.anchor_turn,
                )
            elif mission.kind == "crash_exploit":
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="crash_exploit",
                    eval_turn_fn=lambda turns, desired_arrival=option.anchor_turn: max(turns, desired_arrival),
                    anchor_turn=option.anchor_turn,
                    anchor_tolerance=CRASH_EXPLOIT_ETA_WINDOW,
                )
            else:
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                )
            if plan is None:
                continue

            angle, turns, _, need, send = plan
            if send < need or need > left:
                continue

            sent = append_move(option.src_id, angle, send)
            if sent < need:
                continue
            planned_commitments[target.id].append((turns, world.player, int(sent)))
            continue

        limits = []
        for option in mission.options:
            left = source_attack_left(option.src_id)
            limits.append(min(left, option.send_cap))
        if min(limits) <= 0:
            continue

        missing = world.min_ships_to_own_at(
            target.id,
            mission.turns,
            world.player,
            planned_commitments=planned_commitments,
            upper_bound=sum(limits),
        )
        if missing <= 0 or sum(limits) < missing:
            continue

        ordered = sorted(
            zip(mission.options, limits),
            key=lambda item: (item[0].turns, -item[1], item[0].src_id),
        )
        remaining = missing
        sends = {}
        for idx, (option, limit) in enumerate(ordered):
            remaining_other = sum(other_limit for _, other_limit in ordered[idx + 1 :])
            send = min(limit, max(0, remaining - remaining_other))
            sends[option.src_id] = send
            remaining -= send
        if remaining > 0:
            continue

        reaimed = []
        for option, _ in ordered:
            send = sends.get(option.src_id, 0)
            if send <= 0:
                continue
            src = world.planet_by_id[option.src_id]
            fixed_aim = world.plan_shot(src.id, target.id, send)
            if fixed_aim is None:
                reaimed = []
                break
            angle, turns, _, _ = fixed_aim
            reaimed.append((option.src_id, angle, turns, send))
        if not reaimed:
            continue

        turns_only = [item[2] for item in reaimed]
        eta_tol = swarm_eta_tolerance(mission.options, target, world)
        if max(turns_only) - min(turns_only) > eta_tol:
            continue

        actual_joint_turn = max(turns_only)
        owner_after, _ = world.projected_state(
            target.id,
            actual_joint_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=[(turns, world.player, send) for _, _, turns, send in reaimed],
        )
        if owner_after != world.player:
            continue

        committed = []
        for src_id, angle, turns, send in reaimed:
            actual = append_move(src_id, angle, send)
            if actual <= 0:
                continue
            committed.append((turns, world.player, int(actual)))
        if sum(item[2] for item in committed) < missing:
            continue
        planned_commitments[target.id].extend(committed)

    # Use leftover attack budget for one more pass after the first commitment
    # wave is fixed.
    if not world.is_very_late and allow_optional_phase():
        for src in world.my_planets:
            if expired():
                return finalize_moves()
            src_left = source_attack_left(src.id)
            if src_left < FOLLOWUP_MIN_SHIPS:
                continue

            best = None
            for target in world.planets:
                if expired():
                    return finalize_moves()
                if target.id == src.id or target.owner == world.player:
                    continue
                if target.id in world.comet_ids and target.production <= LOW_VALUE_COMET_PRODUCTION:
                    continue

                seeded = world.best_probe_aim(
                    src.id,
                    target.id,
                    src_left,
                    hints=(int(target.ships) + 1,),
                )
                if seeded is None:
                    continue
                rough_ships, rough_aim = seeded

                est_turns = rough_aim[1]
                if world.is_late and est_turns > world.remaining_steps - LATE_CAPTURE_BUFFER:
                    continue

                rough_needed = world.min_ships_to_own_at(
                    target.id,
                    est_turns,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=src_left,
                )
                if rough_needed <= 0 or rough_needed > src_left:
                    continue
                if opening_filter(target, est_turns, rough_needed, src_left, world, policy):
                    continue

                send = preferred_send(target, rough_needed, est_turns, src_left, world, modes, policy)
                if send < rough_needed:
                    continue

                plan = settle_plan(
                    src,
                    target,
                    src_left,
                    send,
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                )
                if plan is None:
                    continue

                _, turns, _, need, final_send = plan
                if world.is_late and turns > world.remaining_steps - LATE_CAPTURE_BUFFER:
                    continue
                if final_send < need:
                    continue

                value = target_value(target, turns, "capture", world, modes, policy)
                if value <= 0:
                    continue

                score = apply_score_modifiers(
                    value / (final_send + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "capture",
                    world,
                )
                if best is None or score > best[0]:
                    best = (score, target, plan)

            if best is None:
                continue

            _, target, plan = best
            angle, turns, _, need, send = plan
            src_left = source_attack_left(src.id)
            if need > src_left:
                continue

            plan = settle_plan(
                src,
                target,
                src_left,
                min(src_left, send),
                world,
                planned_commitments,
                modes,
                policy,
                mission="capture",
            )
            if plan is None:
                continue

            angle, turns, _, need, send = plan
            if send < need:
                continue

            actual = append_move(src.id, angle, send)
            if actual < need:
                continue
            planned_commitments[target.id].append((turns, world.player, int(actual)))

    # If a planet cannot hold soon, prefer reinforcement first. For stacks that
    # still look doomed after the main mission pass, prefer a last useful
    # capture; otherwise retreat the stack to a safer ally.
    if expired():
        return finalize_moves()
    live_doomed = compute_live_doomed()
    if live_doomed:
        frontier_targets = (
            world.enemy_planets
            if world.enemy_planets
            else (world.static_neutral_planets or world.neutral_planets)
        )
        if frontier_targets:
            frontier_distance = {
                planet.id: nearest_distance_to_set(planet.x, planet.y, frontier_targets)
                for planet in world.my_planets
            }
        else:
            frontier_distance = {planet.id: 10**9 for planet in world.my_planets}

        for planet in world.my_planets:
            if expired():
                return finalize_moves()
            if planet.id not in live_doomed:
                continue

            available_now = source_inventory_left(planet.id)
            if available_now < policy["reserve"].get(planet.id, 0):
                continue

            best_capture = None
            for target in world.planets:
                if expired():
                    return finalize_moves()
                if target.id == planet.id or target.owner == world.player:
                    continue
                seeded = world.best_probe_aim(
                    planet.id,
                    target.id,
                    available_now,
                    hints=(available_now, int(target.ships) + 1),
                )
                if seeded is None:
                    continue
                _, probe_aim = seeded
                probe_turns = probe_aim[1]
                if probe_turns > world.remaining_steps - 2:
                    continue

                need = world.min_ships_to_own_at(
                    target.id,
                    probe_turns,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=available_now,
                )
                if need <= 0 or need > available_now:
                    continue

                plan = settle_plan(
                    planet,
                    target,
                    available_now,
                    min(available_now, max(need, int(target.ships) + 1)),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                )
                if plan is None:
                    continue
                angle, turns, _, plan_need, send = plan
                if send < plan_need:
                    continue
                score = target_value(target, turns, "capture", world, modes, policy) / (send + turns + 1.0)
                if target.owner not in (-1, world.player):
                    score *= 1.05
                if best_capture is None or score > best_capture[0]:
                    best_capture = (score, target.id, angle, turns, send)

            if best_capture is not None:
                _, target_id, angle, turns, need = best_capture
                actual = append_move(planet.id, angle, need)
                if actual >= 1:
                    planned_commitments[target_id].append((turns, world.player, int(actual)))
                continue

            safe_allies = [
                ally
                for ally in world.my_planets
                if ally.id != planet.id and ally.id not in live_doomed
            ]
            if not safe_allies:
                continue

            retreat_target = min(
                safe_allies,
                key=lambda ally: (
                    frontier_distance.get(ally.id, 10**9),
                    planet_distance(planet, ally),
                ),
            )
            aim = world.plan_shot(planet.id, retreat_target.id, available_now)
            if aim is None:
                continue
            angle, _, _, _ = aim
            append_move(planet.id, angle, available_now)

    # Rear planets feed the frontier through staging allies instead of acting
    # as slow solo attackers.
    if (
        (world.enemy_planets or world.neutral_planets)
        and len(world.my_planets) > 1
        and not world.is_late
        and allow_optional_phase()
    ):
        live_doomed = compute_live_doomed()
        frontier_targets = (
            world.enemy_planets
            if world.enemy_planets
            else (world.static_neutral_planets or world.neutral_planets)
        )
        frontier_distance = {
            planet.id: nearest_distance_to_set(planet.x, planet.y, frontier_targets)
            for planet in world.my_planets
        }
        safe_fronts = [
            planet for planet in world.my_planets if planet.id not in live_doomed
        ]
        if safe_fronts:
            front_anchor = min(safe_fronts, key=lambda planet: frontier_distance[planet.id])
            send_ratio = (
                REAR_SEND_RATIO_FOUR_PLAYER if world.is_four_player else REAR_SEND_RATIO_TWO_PLAYER
            )
            if modes["is_finishing"]:
                send_ratio = max(send_ratio, REAR_SEND_RATIO_FOUR_PLAYER)

            for rear in sorted(world.my_planets, key=lambda planet: -frontier_distance[planet.id]):
                if expired():
                    return finalize_moves()
                if rear.id == front_anchor.id or rear.id in live_doomed:
                    continue
                if source_attack_left(rear.id) < REAR_SOURCE_MIN_SHIPS:
                    continue
                if frontier_distance[rear.id] < frontier_distance[front_anchor.id] * REAR_DISTANCE_RATIO:
                    continue

                stage_candidates = [
                    planet
                    for planet in safe_fronts
                    if planet.id != rear.id
                    and frontier_distance[planet.id] < frontier_distance[rear.id] * REAR_STAGE_PROGRESS
                ]
                if stage_candidates:
                    front = min(
                        stage_candidates,
                        key=lambda planet: planet_distance(rear, planet),
                    )
                else:
                    objective = min(
                        frontier_targets,
                        key=lambda target: planet_distance(rear, target),
                    )
                    remaining_fronts = [planet for planet in safe_fronts if planet.id != rear.id]
                    if not remaining_fronts:
                        continue
                    front = min(
                        remaining_fronts,
                        key=lambda planet: planet_distance(planet, objective),
                    )

                if front.id == rear.id:
                    continue

                send = int(source_attack_left(rear.id) * send_ratio)
                if send < REAR_SEND_MIN_SHIPS:
                    continue

                aim = world.plan_shot(rear.id, front.id, send)
                if aim is None:
                    continue

                angle, turns, _, _ = aim
                if turns > REAR_MAX_TRAVEL_TURNS:
                    continue
                append_move(rear.id, angle, send)

    # ---------------------------------------------------------------
    # Domination consolidation: when ahead, send aggressive "total war"
    # fleets at enemy planets to finish them off (Hyperion-style).
    # ---------------------------------------------------------------
    if (
        modes["is_dominating"]
        and modes["domination"] > DOMINATION_CONSOLIDATION_THRESHOLD
        and world.enemy_planets
        and allow_optional_phase()
    ):
        for src in world.my_planets:
            if expired():
                return finalize_moves()
            src_left = source_attack_left(src.id)
            if src_left < 4:
                continue

            best_target = None
            best_score = -1.0
            for target in world.enemy_planets:
                if expired():
                    return finalize_moves()
                if target.owner == world.player:
                    continue

                fleet_size = max(4, int(src_left * LATE_AGGRESSION_FLEET_RATIO))
                aim = world.plan_shot(src.id, target.id, fleet_size)
                if aim is None:
                    continue
                angle, turns, _, _ = aim
                if turns > world.remaining_steps - 2:
                    continue

                need = world.min_ships_to_own_at(
                    target.id,
                    turns,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=fleet_size,
                )
                if need <= 0 or need > fleet_size:
                    continue

                prod_value = target.production * max(1, world.remaining_steps - turns)
                enemy_str = world.owner_strength.get(target.owner, 0)
                elim_bonus = ELIMINATION_BONUS if enemy_str <= WEAK_ENEMY_THRESHOLD else 0.0
                score = (prod_value + elim_bonus) / (need + turns * 0.4 + 1.0)
                if score > best_score:
                    best_score = score
                    best_target = (target, aim, need, fleet_size)

            if best_target is not None:
                target, aim, need, fleet_size = best_target
                angle, turns, _, _ = aim
                send = min(fleet_size, source_attack_left(src.id))
                if send >= need:
                    actual = append_move(src.id, angle, send)
                    if actual >= need:
                        planned_commitments[target.id].append((turns, world.player, int(actual)))

    return finalize_moves()