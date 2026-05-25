import math
import kaggle_environments.envs.orbit_wars.orbit_wars as ow
steps = 0
moving_planets = []
planets_coords = {}
fleet_trajectories = []
reinforcement_trajectories = []
MAX_SPEED = 6.0
MIN_SHIPS_MY_PLANET_ATTACK = 5
MIN_SHIPS_TARGET_COOP_ATTACK = 20
COOP_PLANET_CAP = 8
FORMULA_DIST = 100
FORMULA_PROD_MULT = 15
FORMULA_ENEMY_BONUS_MULT = 10
FORMULA_TOTAL_SHIPS_PERCENT = 0.7
def get_custom_score(my_planet, target):
    dist = math.sqrt((my_planet.x - target.x)**2 + (my_planet.y - target.y)**2)
    min_ships = target.ships + 1
    fleet_speed = get_fleet_speed(max(1, min_ships))
    eta = dist / fleet_speed
    enemy_producted = 0
    enemy_bonus = 0
    if target.owner != -1:
        enemy_producted = eta * target.production
        enemy_bonus = target.production
    
    total_ships = min_ships + enemy_producted
    return (
        (FORMULA_DIST - dist)
        + (FORMULA_PROD_MULT * target.production)
        + (FORMULA_ENEMY_BONUS_MULT * enemy_bonus)
        - (FORMULA_TOTAL_SHIPS_PERCENT * total_ships)
        - (2 * eta)
    )
def refresh_local_obs(obs) -> dict:
    planets = [ow.Planet(*p) for p in obs.get("planets", [])]
    my_planets = [p for p in planets if p.owner == obs.get("player", [])]
    targets = [p for p in planets if p.owner != obs.get("player", [])]
    player = obs.get("player", -2)
    fleets = [ow.Fleet(*p) for p in obs.get("fleets", [])]
    return {
        "planets": planets,
        "my_planets": my_planets,
        "targets": targets,
        "player": player,
        "fleets": fleets
    }
def fill_moving_planets(obs):
    planets = [ow.Planet(*p) for p in obs.get("planets", [])]
    initial_by_id = {i[0]: ow.Planet(*i) for i in obs.get("initial_planets", [])}
    for p in planets:
        i = initial_by_id[p.id]
        if (p.x, p.y) != (i.x, i.y):
            if p.id not in moving_planets:
                moving_planets.append(p.id)
def get_planet_trajectories(planet, velocity):
    planet_trajectories = []
    angle = math.atan2(planet.y - 50, planet.x - 50)
    r = math.sqrt((planet.x - 50)**2 + (planet.y - 50)**2)
    for tick in range(1, 61):
        angle_t = angle + velocity * tick
        x_t = 50 + r * math.cos(angle_t)
        y_t = 50 + r * math.sin(angle_t)
        planet_trajectories.append((x_t, y_t))
    
    return planet_trajectories
def update_fleet_trajectories(fleets):
    for f_t in fleet_trajectories[:]:
        found = False
        for f in fleets:
            if f.from_planet_id == f_t["my_planet"].id and abs(f.angle - f_t["angle"]) < 1e-6:
                found = True
                break
        
        if found:
            f_t["arrive_tick"] = max(0, f_t["arrive_tick"] - 1)
        
        if not found:
            fleet_trajectories.remove(f_t)
def update_reinforcement_trajectories(planets):
    planet_ids = {p.id for p in planets}
    for r_t in reinforcement_trajectories[:]:
        r_t["arrive_tick"] -= 1
        if r_t["arrive_tick"] <= 0:
            reinforcement_trajectories.remove(r_t)
            continue
def get_planet_under_attack(my_planets, fleets, player, angular_velocity):
    moving_planet_trajectories = {}
    planets_under_attack = {}
    seen = set()
    for my_planet in my_planets:
        if my_planet.id in moving_planets:
            moving_planet_trajectories[my_planet.id] = get_planet_trajectories(my_planet, angular_velocity)
    fleets = [f for f in fleets if f.owner != player]
    for fleet in fleets:
        fleet_speed = get_fleet_speed(fleet.ships)
        prev_x = fleet.x
        prev_y = fleet.y
        for tick in range(1, 61):
            next_x = fleet.x + math.cos(fleet.angle) * fleet_speed * tick
            next_y = fleet.y + math.sin(fleet.angle) * fleet_speed * tick
            for my_planet in my_planets:
                if my_planet.id in moving_planets:
                    m_x, m_y = moving_planet_trajectories[my_planet.id][tick-1]
                else:
                    m_x, m_y = my_planet.x, my_planet.y
                if collides(prev_x, prev_y, next_x, next_y, m_x, m_y, my_planet.radius):
                    if (my_planet.id, fleet.id) not in seen:
                        if my_planet.id not in planets_under_attack:
                            planets_under_attack[my_planet.id] = {
                                "planet": my_planet,
                                "fleets": []
                            }
                        planets_under_attack[my_planet.id]["fleets"].append({
                            "fleet": fleet,
                            "arrive_tick": tick
                        })
                        seen.add((my_planet.id, fleet.id))
        
            prev_x, prev_y = next_x, next_y
    return planets_under_attack
def get_closest_planets_to_target(my_planets, target):
    planets = []
    for my_planet in my_planets:
        dist = math.sqrt((my_planet.x - target.x)**2 + (my_planet.y - target.y)**2)
        planets.append((my_planet, dist))
    planets.sort(key = lambda k: k[1])
    return planets
def collides(x1, y1, x2, y2, cx, cy, r) -> bool:
    vec_x = x2 - x1
    vec_y = y2 - y1
    vec_to_cx = cx - x1
    vec_to_cy = cy - y1
    vec_length_sq = vec_x**2 + vec_y**2
    if vec_length_sq == 0:
        dx = x1 - cx
        dy = y1 - cy
        return dx**2 + dy**2 <= r**2
    closest_point = (vec_to_cx * vec_x + vec_to_cy * vec_y) / vec_length_sq
    closest_point = max(0, min(1, closest_point))
    closest_x = x1 + closest_point * vec_x
    closest_y = y1 + closest_point * vec_y
    dx = closest_x - cx
    dy = closest_y - cy
    return dx**2 + dy**2 <= r**2
def check_sun_collision(m, fleet_speed, angle, ticks=61) -> bool:
    prev_x, prev_y = m.x, m.y
    for tick in range(1, ticks):
        x = m.x + fleet_speed * math.cos(angle) * tick
        y = m.y + fleet_speed * math.sin(angle) * tick
        if collides(prev_x, prev_y, x, y, 50, 50, 10):
            return True
        prev_x, prev_y = x, y
    return False
def get_fleet_speed(ships):
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
def calculate_angle(f, t):
    return math.atan2(t.y-f.y, t.x-f.x)
def get_planet_pos_at_tick(p, velocity, tick, is_moving):
    if not is_moving:
        return p.x, p.y
    
    angle = math.atan2(p.y - 50, p.x - 50)
    r = math.sqrt((p.x - 50)**2 + (p.y - 50)**2)
    angle_t = angle + velocity * tick
    x_t = 50 + r * math.cos(angle_t)
    y_t = 50 + r * math.sin(angle_t)
    
    return x_t, y_t

def check_path_collision(f, t, fleet_speed, angle, ticks, all_planets, velocity):
    prev_x, prev_y = f.x, f.y
    
    for tick in range(1, int(ticks) + 1):
        x = f.x + fleet_speed * math.cos(angle) * tick
        y = f.y + fleet_speed * math.sin(angle) * tick

        if collides(prev_x, prev_y, x, y, 50, 50, 10):
            return True

        for p in all_planets:
            if p.id == f.id or p.id == t.id:
                continue
            
            is_moving = p.id in moving_planets
            
            px, py = get_planet_pos_at_tick(p, velocity, tick, is_moving)
            
            if collides(prev_x, prev_y, x, y, px, py, p.radius):
                return True

        prev_x, prev_y = x, y
        
    return False

def find_angle_to_planet(f, t, ships, velocity, all_planets, moving=False):
    fleet_speed = get_fleet_speed(ships)
    if moving:
        planets_trajectories = get_planet_trajectories(t, velocity)
        for tick, (tx, ty) in enumerate(planets_trajectories, start=1):
            dx = tx - f.x
            dy = ty - f.y
            dist_to_target = math.sqrt(dx**2 + dy**2) - f.radius # tru di t.radius nua thi sao
            travel_dist = fleet_speed * tick
            miss_dist = abs(dist_to_target - travel_dist)
            if miss_dist > t.radius:
                continue
            angle = math.atan2(dy, dx)
            if check_sun_collision(f, fleet_speed, angle):
                return None, None
            
            return angle, tick
        return None, None
    else:
        angle = calculate_angle(f, t)
        if check_sun_collision(f, fleet_speed, angle):
            return None, None
        dist = math.sqrt((f.x - t.x)**2 + (f.y-t.y)**2)
        tick = math.floor(dist / fleet_speed)
        return angle, tick
    return None, None
def get_reinforcement_plans(my_planets, planets_under_attack):
    
    reinforcement_plans = {}
    
    for my_planet in my_planets:
        if my_planet.id in planets_under_attack:
            attacking_fleets = sorted(
                planets_under_attack[my_planet.id]["fleets"],
                key=lambda k: k["arrive_tick"]
            )
    
            incoming_reinforcements = sorted(
                [r for r in reinforcement_trajectories if r["target"].id == my_planet.id],
                key=lambda k: k["arrive_tick"]
            )
            my_planet_available_ships = my_planet.ships
            previous_tick = 0
            r_idx = 0
            for fleet in attacking_fleets:
                fleet_arrive_tick = fleet["arrive_tick"]
                my_planet_available_ships += (fleet_arrive_tick - previous_tick) * my_planet.production
                while (
                    r_idx < len(incoming_reinforcements)
                    and incoming_reinforcements[r_idx]["arrive_tick"] <= fleet_arrive_tick
                ):
                    my_planet_available_ships += incoming_reinforcements[r_idx]["ships"]
                    r_idx += 1
                
                enemy_ships = fleet["fleet"].ships
                my_planet_available_ships -= enemy_ships
                previous_tick = fleet_arrive_tick
                if my_planet_available_ships < 0:
                    reinforcement_needed = max(MIN_SHIPS_MY_PLANET_ATTACK, abs(my_planet_available_ships))
                    reinforcement_plans[my_planet] = {
                        "ships_needed": reinforcement_needed,
                        "needed_by_tick": fleet_arrive_tick
                    }
                    break
    return reinforcement_plans
def get_candidate_targets(my_planet, targets, comet_planet_ids):
    candidate_targets = []
    for target in targets:
        if target.id in comet_planet_ids:
            continue
        score = get_custom_score(my_planet, target)
        candidate_targets.append((my_planet, target, score))
    
    return sorted(candidate_targets, key=lambda k: k[2], reverse=True)
def predict_total_ships(my_planet, target, velocity, all_planets, base_ships, available_ships, target_is_moving=False):
    total_ships = base_ships
    for _ in range(5):
        angle, arrive_tick = find_angle_to_planet(my_planet, target, total_ships, velocity, all_planets, moving=target_is_moving)
        
        if angle is None:
            return None, None, None
        
        if target.owner != -1:
            new_total_ships = base_ships + arrive_tick * target.production
        else:
            new_total_ships = base_ships
        if new_total_ships > available_ships:
            return None, None, None
        if new_total_ships == total_ships:
            break
        
        total_ships = new_total_ships
    return total_ships, angle, arrive_tick
def plan_coop_attack(attacking_planets, target, base_ships, velocity, all_planets, target_is_moving=False):
    remainder = base_ships
    planned = []
    for attacking_planet in attacking_planets:
        plannet = attacking_planet["planet"]
        ships = min(attacking_planet["ships"], remainder)
        if ships > 0:
            ships = min(attacking_planet["ships"], max(ships, MIN_SHIPS_MY_PLANET_ATTACK))
        if ships <= 0:
            continue
        angle, arrive_tick = find_angle_to_planet(plannet, target, ships, velocity, all_planets, target_is_moving)
        remainder -= ships
        if angle is None or arrive_tick is None:
            continue
            
        planned.append([plannet, angle, ships, arrive_tick])
    return remainder, planned
def nctrinh(obs) -> list:
    global steps
    global fleet_trajectories
    global reinforcement_trajectories
    moves = []
    if steps < 2:
        steps += 1
        return []
    
    if steps == 2:
        fill_moving_planets(obs)
        steps = 3
    lobs = refresh_local_obs(obs)
    update_fleet_trajectories(lobs.get("fleets", []))
    update_reinforcement_trajectories(lobs.get("planets", []))
    comet_planet_ids = obs.get("comet_planet_ids", [])
    planets_under_attack = get_planet_under_attack(
                            lobs.get("my_planets", []), 
                            lobs.get("fleets", []),
                            lobs.get("player", -2),
                            obs.angular_velocity)
    exhausted_planets_id = set()
    if not lobs.get("targets", []):
        return []
    # =====================================================================
    # 1. HỖ TRỢ CHI VIỆN (Phòng thủ các hành tinh có thể cứu)
    # =====================================================================
    reinforcement_plans = get_reinforcement_plans(lobs.get("my_planets", []), planets_under_attack)
    for my_planet, plan in reinforcement_plans.items():
        already_reinforced = any(
            r["target"].id == my_planet.id and r["arrive_tick"] >= 0
            for r in reinforcement_trajectories
        )
        if already_reinforced:
            continue
        ships_needed = plan["ships_needed"]
        needed_by_tick = plan["needed_by_tick"]
        nearest_planets = get_closest_planets_to_target(lobs.get("my_planets", []), my_planet)
        for nearest_planet_and_dist in nearest_planets:
            nearest_planet, _ = nearest_planet_and_dist
            if nearest_planet.id == my_planet.id or nearest_planet.id in exhausted_planets_id:
                continue
            nearest_planet_available_ships = nearest_planet.ships
            reserved_reinforcement_ships = sum(
                r["ships"]
                for r in reinforcement_trajectories
                if r["my_planet"].id == nearest_planet.id
            )
            nearest_planet_available_ships -= reserved_reinforcement_ships
            if nearest_planet.id in planets_under_attack:
                enemy_ships = sum(
                    planet_under_attack['fleet'].ships
                    for planet_under_attack in planets_under_attack[nearest_planet.id]["fleets"]
                )
                nearest_planet_available_ships = max(0, nearest_planet_available_ships - enemy_ships)
            sent_reinforcements = max(MIN_SHIPS_MY_PLANET_ATTACK, ships_needed)
            if nearest_planet_available_ships < sent_reinforcements:
                continue
            angle_nearest_planet = None
            if my_planet.id in moving_planets:
                planet_is_moving = True
            else:
                planet_is_moving = False
            angle_nearest_planet, arrive_tick = find_angle_to_planet(nearest_planet, my_planet, sent_reinforcements, obs.angular_velocity, lobs.get("planets", []), planet_is_moving)
            if (
                angle_nearest_planet is None
                or arrive_tick is None
                or arrive_tick > needed_by_tick
            ):
                continue
            moves.append([nearest_planet.id, angle_nearest_planet, sent_reinforcements])
            exhausted_planets_id.add(nearest_planet.id)
            reinforcement_trajectories.append({
                "my_planet": nearest_planet, 
                "target": my_planet, 
                "angle": angle_nearest_planet,
                "ships": sent_reinforcements,
                "arrive_tick": arrive_tick
            })
            break
    # =====================================================================
    # 2. CHIẾN THUẬT ĐỔI NHÀ & VƯỜN KHÔNG NHÀ TRỐNG (Hành tinh thất thủ)
    # =====================================================================
    for my_planet in lobs.get("my_planets", []):
        if my_planet.id in exhausted_planets_id or my_planet.ships <= 0:
            continue

        # Đánh giá xem hành tinh có chắc chắn bị chiếm không
        is_doomed = False
        if my_planet.id in planets_under_attack:
            enemy_ships = sum(f["fleet"].ships for f in planets_under_attack[my_planet.id]["fleets"])
            incoming_reinf = sum(r["ships"] for r in reinforcement_trajectories if r["target"].id == my_planet.id)
            
            if my_planet.ships + incoming_reinf < enemy_ships:
                is_doomed = True

        if is_doomed:
            evacuated = False
            evac_ships = my_planet.ships # Lấy TOÀN BỘ quân để sơ tán/đổi nhà
            
            # Ưu tiên 1: ĐỔI NHÀ - Tìm mục tiêu yếu để chiếm
            candidate_targets = get_candidate_targets(my_planet, lobs.get("targets", []), comet_planet_ids)
            for _, target, _ in candidate_targets:
                ships_needed = target.ships + 1
                if target.owner != -1:
                    ships_needed += 3 * target.production
                
                if evac_ships >= ships_needed:
                    target_is_moving = target.id in moving_planets
                    # Dự đoán xem số quân sơ tán có bay tới an toàn không
                    _, angle, arrive_tick = predict_total_ships(
                        my_planet, target, obs.angular_velocity, lobs.get("planets", []), 
                        ships_needed, evac_ships, target_is_moving
                    )
                    
                    if angle is not None and arrive_tick is not None and not check_sun_collision(my_planet, get_fleet_speed(evac_ships), angle, arrive_tick):
                        moves.append([my_planet.id, angle, evac_ships]) 
                        exhausted_planets_id.add(my_planet.id)
                        fleet_trajectories.append({
                            "my_planet": my_planet, "target": target, "angle": angle,
                            "ships": evac_ships, "arrive_tick": arrive_tick
                        })
                        evacuated = True
                        break 
            
            # Ưu tiên 2: VƯỜN KHÔNG NHÀ TRỐNG - Nếu không thể đánh, chạy sang đồng minh
            if not evacuated:
                safe_allies = []
                for ally in lobs.get("my_planets", []):
                    if ally.id == my_planet.id: 
                        continue
                    
                    ally_enemy_ships = 0
                    if ally.id in planets_under_attack:
                        ally_enemy_ships = sum(f["fleet"].ships for f in planets_under_attack[ally.id]["fleets"])
                    
                    ally_incoming_reinf = sum(r["ships"] for r in reinforcement_trajectories if r["target"].id == ally.id)
                    
                    # Chỉ chạy sang hành tinh đồng minh nào đang an toàn
                    if ally.ships + ally_incoming_reinf >= ally_enemy_ships:
                        dist = math.sqrt((my_planet.x - ally.x)**2 + (my_planet.y - ally.y)**2)
                        safe_allies.append((ally, dist))
                
                safe_allies.sort(key=lambda k: k[1]) 
                
                for ally, dist in safe_allies:
                    ally_is_moving = ally.id in moving_planets
                    angle, arrive_tick = find_angle_to_planet(
                        my_planet, ally, evac_ships, obs.angular_velocity, 
                        lobs.get("planets", []), ally_is_moving
                    )
                    
                    if angle is not None and arrive_tick is not None and not check_sun_collision(my_planet, get_fleet_speed(evac_ships), angle, arrive_tick):
                        moves.append([my_planet.id, angle, evac_ships]) 
                        exhausted_planets_id.add(my_planet.id)
                        reinforcement_trajectories.append({
                            "my_planet": my_planet, "target": ally, "angle": angle,
                            "ships": evac_ships, "arrive_tick": arrive_tick
                        })
                        evacuated = True
                        break

            exhausted_planets_id.add(my_planet.id)

    for my_planet in sorted(
            lobs.get("my_planets", []), 
            key=lambda k: k.ships,
            reverse=True
        ):
        if my_planet.id in exhausted_planets_id or my_planet.ships < MIN_SHIPS_MY_PLANET_ATTACK:
            continue
        candidate_targets = get_candidate_targets(my_planet, lobs.get("targets", []), comet_planet_ids)
        for planet, target, score in candidate_targets[:3]:
            my_planet_available_ships = planet.ships
            if planet.id in planets_under_attack:
                enemy_ships = sum(
                    fleet["fleet"].ships
                    for fleet in planets_under_attack[planet.id]["fleets"]
                )
                my_planet_available_ships = max(0, planet.ships - enemy_ships)
            if my_planet_available_ships < MIN_SHIPS_MY_PLANET_ATTACK:
                continue
            nearest_planets = get_closest_planets_to_target(lobs.get("my_planets", []), target)
            safe_nearest_planets = []
            for nearest_planet, dist in nearest_planets:
                if nearest_planet.id == planet.id or nearest_planet.id in exhausted_planets_id:
                    continue
                available_ships = nearest_planet.ships
                if nearest_planet.id in planets_under_attack:
                    enemy_ships = sum(
                        fleet["fleet"].ships 
                        for fleet in planets_under_attack[nearest_planet.id]["fleets"]
                    )
                    available_ships = max(0, nearest_planet.ships - enemy_ships)
    
                if available_ships < MIN_SHIPS_MY_PLANET_ATTACK:
                    continue
                
                safe_nearest_planets.append((nearest_planet, dist, available_ships))
            
            owned_count = len(lobs.get("my_planets", []))
            total_count = len(lobs.get("planets", []))
            en_route = 0
            if fleet_trajectories:
                en_route = sum(
                    fleet["ships"]
                    for fleet in fleet_trajectories
                    if fleet["target"].id == target.id
                )
            
            ships_needed = target.ships + 1
            if target.owner != -1:
                ships_needed += 3 * target.production
            
            if owned_count < total_count * 0.75:
                if en_route >= ships_needed:
                    continue
            base_ships = max(MIN_SHIPS_MY_PLANET_ATTACK, ships_needed - en_route)
            if my_planet_available_ships >= base_ships:
                if target.id in moving_planets:
                    target_is_moving = True
                    
                else:
                    target_is_moving = False
                total_ships, angle, arrive_tick = predict_total_ships(
                        planet, 
                        target, 
                        obs.angular_velocity, 
                        lobs.get("planets", []), 
                        base_ships, 
                        my_planet_available_ships, 
                        target_is_moving
                    )                    
                
                if angle is not None and arrive_tick is not None:
                    fleet_speed = get_fleet_speed(max(1, total_ships))
                    collides_sun = check_sun_collision(planet, fleet_speed, angle)
                    if collides_sun:
                        continue
                    moves.append([planet.id, angle, total_ships])
                    exhausted_planets_id.add(planet.id)
                    fleet_trajectories.append({
                            "my_planet": planet,
                            "target": target,
                            "angle": angle,
                            "ships": total_ships,
                            "arrive_tick": arrive_tick
                        })
            elif my_planet_available_ships < base_ships and len(lobs.get("my_planets", [])) > 1 and target.ships >= MIN_SHIPS_TARGET_COOP_ATTACK:
                accum = my_planet_available_ships
                attacking_planets = [{"planet": planet, "ships": my_planet_available_ships}]
                coop_sent = False
                for safe_nearest_planet, dist, safe_nearest_available_ships in safe_nearest_planets:
                    if coop_sent:
                        break
                    attacking_planets.append({"planet": safe_nearest_planet, "ships": safe_nearest_available_ships})
                    accum += safe_nearest_available_ships
                    if len(attacking_planets) > COOP_PLANET_CAP:
                        break
                    if accum < base_ships:
                        continue
                    
                    if target.id in moving_planets:
                        target_is_moving = True
                        
                    else:
                        target_is_moving = False
                    remainder, planned = plan_coop_attack(
                            attacking_planets,
                            target,
                            base_ships,
                            obs.angular_velocity,
                            lobs.get("planets", []), 
                            target_is_moving
                    )
                    if remainder > 0:
                        continue
                    for move in planned:
                        fleet_trajectories.append({
                            "my_planet": move[0],
                            "target": target,
                            "angle": move[1],
                            "ships": move[2],
                            "arrive_tick": move[3]
                        })
                        exhausted_planets_id.add(move[0].id)
                        move[0] = move[0].id
                        moves.append(move)
                    coop_sent = True
                    break
    return moves