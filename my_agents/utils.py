import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

def static_collision_check(x0, y0, angle, v, starting_id, planets, orbiting_planets):
    # with u the vector unit of the fleet's direction,
    # and v the fleet's speed, we have the fleet's position at time t as:
    # x = x0 + u_x * t * v
    # y = y0 + u_y * t * v
    # we want to check if at any point the fleet is within the radius of a planet, solve
    # (x - p.x)^2 + (y - p.y)^2 < p.radius^2
    # which is x^2 - 2*x*p.x + p.x^2 + y^2 - 2*y*p.y + p.y^2 < p.radius^2
    # reduced into a quadratic equation of t:
    # t^2*v^2(u_x^2 + u_y^2) - 2*t*v(u_x(p.x - x0) + u_y(p.y - y0)) + (x0 - p.x)^2 + (y0 - p.y)^2 - p.radius^2 < 0

    col_list = []

    u_x = math.cos(angle)
    u_y = math.sin(angle)

    A = v**2 * (u_x**2 + u_y**2)

    for p in planets:
        if p.id in orbiting_planets or p.id == starting_id:
            continue
        B = -2 * v * (u_x * (p.x - x0) + u_y * (p.y - y0))
        C = (x0 - p.x)**2 + (y0 - p.y)**2 - p.radius**2
        delta = B**2 - 4 * A * C
        
        if delta <= 0:
            continue
        else:
            # solve for a smallest positive t
            t1 = (-B - math.sqrt(delta)) / (2 * A)
            t2 = (-B + math.sqrt(delta)) / (2 * A)
            if t1 > 0 and t2 > 0:
                col_list.append((p.id, min(t1, t2)))
            elif t1 > 0:
                col_list.append((p.id, t1))
            elif t2 > 0:
                col_list.append((p.id, t2))

    if len(col_list) > 0:
        col_list.sort(key=lambda x: x[1])
        return col_list[0][0], col_list[0][1]
    else:
        return None
    
def orbiting_collision_check(x0, y0, angle, v, starting_id, planets, orbiting_planets, comets, angular_velocity, max_turns):
    # check collision once every while, as solving the equation is kinda troubling
    # to handle small planets, we check once every half a turn
    v_prime = v / 2
    angular_velocity_prime = angular_velocity / 2
    for t in range(1, 2 * max_turns + 1):
        fleet_x = x0 + v_prime * t * math.cos(angle)
        fleet_y = y0 + v_prime * t * math.sin(angle)

        for p in planets:
            if p.id not in orbiting_planets or p.id in comets or p.id == starting_id:
                continue
            phi = math.atan2(p.y - 50, p.x - 50)
            orbit_r = orbiting_planets[p.id]
            planet_x = 50 + orbit_r * math.cos(phi + angular_velocity_prime * t)
            planet_y = 50 + orbit_r * math.sin(phi + angular_velocity_prime * t)

            if math.hypot(fleet_x - planet_x, fleet_y - planet_y) < p.radius:
                return p.id, t * 0.5 # earliest collision

    return None

def compute_move_angle(op, tp, send, planets, orbiting_planets, comets, angular_velocity, max_turns):
    is_orbiting = tp.id in orbiting_planets
    dx = tp.x - op.x
    dy = tp.y - op.y
    fleet_speed = 1.0 + 5.0 * (math.log(send) / math.log(1000)) ** 1.5

    if not is_orbiting:
        angle = math.atan2(dy, dx)
        distance = math.hypot(dx, dy) - op.radius - tp.radius
        turns_needed = distance / fleet_speed
        x0 = op.x + op.radius * math.cos(angle)
        y0 = op.y + op.radius * math.sin(angle)
        static_result = static_collision_check(x0, y0, angle, fleet_speed, op.id, planets, orbiting_planets)
        dynamic_result = orbiting_collision_check(x0, y0, angle, fleet_speed, op.id, planets, orbiting_planets, comets, angular_velocity, max_turns)
        if static_result is None:
            return None
        elif static_result[0] != tp.id:
            return None
        elif dynamic_result is not None:
            if dynamic_result[1] < turns_needed:
                return None
            
        sun = Planet(-1, -1, 50, 50, 10, 0, 0)
        sun_result = static_collision_check(x0, y0, angle, fleet_speed, op.id, [sun], orbiting_planets)
        if sun_result is not None and sun_result[1] < turns_needed:
            return None

        return angle, turns_needed # the more precise the better, no need to round

    else:
        phi = math.atan2(tp.y - 50, tp.x - 50)
        orbit_r = orbiting_planets[tp.id]
        angular_velocity_prime = angular_velocity / 2
        fleet_speed_prime = fleet_speed / 2
        for t in range(1, 2 * max_turns + 1):
            new_tp_x = 50 + orbit_r * math.cos(phi + angular_velocity_prime * t)
            new_tp_y = 50 + orbit_r * math.sin(phi + angular_velocity_prime * t)

            angle = math.atan2(new_tp_y - op.y, new_tp_x - op.x)

            fleet_starting_x = op.x + op.radius * math.cos(angle)
            fleet_starting_y = op.y + op.radius * math.sin(angle)
            new_fleet_x = fleet_starting_x + fleet_speed_prime * t * math.cos(angle)
            new_fleet_y = fleet_starting_y + fleet_speed_prime * t * math.sin(angle)

            if math.hypot(new_fleet_x - new_tp_x, new_fleet_y - new_tp_y) < tp.radius:
                # if the fleet hit the sun first
                sun = Planet(-1, -1, 50, 50, 10, 0, 0)
                sun_result = static_collision_check(fleet_starting_x, fleet_starting_y, angle, fleet_speed, op.id, [sun], orbiting_planets)
                if sun_result is not None and sun_result[1] * 2 < t:
                    return None
                
                # other planets
                static_result = static_collision_check(fleet_starting_x, fleet_starting_y, angle, fleet_speed, op.id, planets, orbiting_planets)
                if static_result is not None and static_result[1] * 2 < t:
                    return None
                dynamic_result = orbiting_collision_check(fleet_starting_x, fleet_starting_y, angle, fleet_speed, op.id, planets, orbiting_planets, comets, angular_velocity, max_turns)
                # always hit since we found a t for the tp
                if dynamic_result[1] * 2 < t:
                    return None

                return angle, t / 2

    return None