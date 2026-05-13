import xr

# 1. Create an OpenXR API instance
instance_create_info = xr.InstanceCreateInfo()
instance_create_info.application_info.application_name = b"Get Device Name"
instance_create_info.application_info.api_version = xr.Version(1, 0, 0)
instance = xr.create_instance(instance_create_info)

# 2. Get the system ID for the HMD
system_get_info = xr.SystemGetInfo()
system_get_info.form_factor = xr.FormFactor.HEAD_MOUNTED_DISPLAY
system_id = xr.get_system(instance, system_get_info)

# 3. Get the system properties
system_properties = xr.get_system_properties(instance, system_id)

# 4. Print the device name
print(f"Device Name: {system_properties.system_name.decode()}")

# 5. Clean up
xr.destroy_instance(instance)