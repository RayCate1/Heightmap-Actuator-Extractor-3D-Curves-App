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

   # Example usage:
   vx = 3  # x-component of velocity
   vy = 4  # y-component of velocity
   angle_rad = calculate_velocity_angle(vx, vy)
   angle_deg = math.degrees(angle_rad)  # Convert to degrees if needed

   print(f"Angle in radians: {angle_rad}")
   print(f"Angle in degrees: {angle_deg}")
