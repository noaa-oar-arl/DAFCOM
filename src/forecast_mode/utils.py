#===========
#import library
#=============



#=============
#main
#==============


class Util
    def regridder(self, config):
        target_file = os.path.expandvars(config['analysis']['target_grid'])
        ds_target = xr.open_dataset(target_file)

        regridder_dict = dict()

        for name in config[config_group]:
            base_file = os.path.expandvars(config[config_group][name]['regrid']['base_grid'])
            ds_base = xr.open_dataset(base_file)
            method = config[config_group][name]['regrid']['method']
            regridder = xe.Regridder(ds_base, ds_target, method)
            regridder_dict[name] = regridder

        return regridder_dict
