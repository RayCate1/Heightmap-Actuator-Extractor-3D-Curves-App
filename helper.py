   def calculate_velocity_angle(vx, vy):
       """
       Calculates the angle of a velocity vector in radians.

       Args:
           vx: The x-component of the velocity vector.
           vy: The y-component of the velocity vector.

       Returns:
           The angle of the velocity vector in radians.
       """
       angle_rad = math.atan2(vy, vx)
       return angle_rad
