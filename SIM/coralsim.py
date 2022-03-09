#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains classes and methods required to set up larval connectivity
experiments in OceanParcels.

@author: Noam Vogt-Vincent
@year: 2022

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import warnings
from parcels import (Field, FieldSet, ParticleSet, JITParticle, AdvectionRK4,
                     ErrorCode, Geographic, GeographicPolar, Variable,
                     DiffusionUniformKh, ParcelsRandom)
from netCDF4 import Dataset, num2date
from datetime import timedelta, datetime



class experiment():
    """
    Initialise a larval dispersal experiment.
    -----------
    Functions:
        import_grid
        import_currents
        set_release_time:
        set_release_loc:
        set_release_params:
        set_partitions:
        set_
    """

    def __init__(self, root_dir):
        # Set up a status dictionary so we know the completion status of the
        # experiment configuration

        self.status = {'grid': False,
                       'currents': False,
                       'release_loc': False,
                       'release_time': False,
                       'partitions': False}

        # Set up dictionaries for various parameters
        self.fh = {}
        self.params = {}

        self.root_dir = root_dir

    def import_grid(self, **kwargs):
        """

        Parameters (* are required)
        ----------
        kwargs :
            fh* : File handle to grid netcdf file (if not already passed)
            grid_type* : 'C' or 'A' (C-grid or A-grid')
            dimensions* : Dict containing dimension names for the grid in the
                          following format:

                              A-GRID:
                              {'lon': LON_NAME,
                               'lat': LAT_NAME}

                              C-GRID:
                              {'rho': {'lon': LON_RHO_NAME,
                                       'lat': LAT_RHO_NAME},
                               'psi': {'lon': LON_PSI_NAME,
                                       'lat': LAT_PSI_NAME}}

        """

        if not all (key in kwargs.keys() for key in ['grid_type', 'dimensions']):
            raise KeyError('Please make sure all required arguments are passed')
        elif 'grid' not in self.fh.keys() and 'fh' in kwargs.keys():
            self.fh['grid'] = kwargs['fh']
        elif 'grid' not in self.fh.keys():
            raise KeyError('Please make sure all required arguments are passed')

        # Check that grid types and dimensions are supplied correctly
        if kwargs['grid_type'] in ['C', 'A']:
            self.params['grid_type'] = kwargs['grid_type']
            if self.params['grid_type'] == 'C':
                # Import C-Grid dimension names
                raise NotImplementedError('C-grids have not been implemented yet!')
            else:
                # Import A-Grid dimension names
                if all (key in kwargs['dimensions'].keys() for key in ['rho', 'psi']):
                    self.params['dimension_names'] = kwargs['dimensions']
                elif all (key in kwargs['dimensions'].keys() for key in ['lon', 'lat']):
                    raise NotImplementedError('Psi grids must currently be present in grid file')
                    self.params['dimension_names'] = {'rho': kwargs['dimensions']}
                else:
                    raise Exception(('Please supply the \'lon\' and \'lat\' '
                                     'dimension names as a dictionary'))
        else:
            raise Exception('Grid type must be \'C\' or \'A\'')

        # Import coordinates for grids
        if self.params['grid_type'] == 'A':
            with Dataset(self.fh['grid'], mode='r') as nc:
                self.axes = {'rho': {},
                             'psi': {}}
                self.axes['rho']['lon'] = np.array(nc.variables[self.params['dimension_names']['rho']['lon']][:])
                self.axes['rho']['lat'] = np.array(nc.variables[self.params['dimension_names']['rho']['lat']][:])
                self.axes['rho']['nx'] = len(self.axes['rho']['lon'])
                self.axes['rho']['ny'] = len(self.axes['rho']['lat'])

                self.axes['psi']['lon'] = np.array(nc.variables[self.params['dimension_names']['psi']['lon']][:])
                self.axes['psi']['lat'] = np.array(nc.variables[self.params['dimension_names']['psi']['lat']][:])
                self.axes['psi']['nx'] = len(self.axes['psi']['lon'])
                self.axes['psi']['ny'] = len(self.axes['psi']['lat'])

        self.status['grid'] = True

    def import_currents(self, **kwargs):
        """

        Parameters (* are required)
        ----------
        kwargs :
            fh* : File handle/s to current data
            variables* : Dictionary of variable names (in OceanParcels format)
            dimensions* : Dictionary of dimension names (in OceanParcels format)
            interp_method: Parcels interpolation method, otherwise defaults for
                           A-Grid ('linear') and C-Grid ('cgrid_velocity')

        """

        if not all (key in kwargs.keys() for key in ['fh', 'variables', 'dimensions']):
            raise KeyError('Please make sure all required arguments are passed')
        else:
            self.fh['currents'] = kwargs['fh']

        if 'interp_method' in kwargs:
            interp_method = kwargs['interp_method']
        elif self.status['grid']:
            if self.params['grid_type'] == 'A':
                interp_method = 'linear'
            else:
                interp_method = 'cgrid_velocity'
        else:
            raise Exception('Either specify an interp method, or run import_grid first.')

        # Create the velocity fieldset
        self.fieldset = FieldSet.from_netcdf(filenames=kwargs['fh'],
                                             variables=kwargs['variables'],
                                             dimensions={'U': kwargs['dimensions'],
                                                         'V': kwargs['dimensions']},
                                             interp_method={'U': interp_method,
                                                            'V': interp_method})

        self.status['currents'] = True

    def create_particles(self, **kwargs):
        """
        Parameters (* are required)
        ----------
        kwargs :
            fh : File handle to grid netcdf file (if not already present)
            num_per_cell : Number of particles to (aim to) initialise per cell
            eez_filter: List of EEZs to release particles from
            eez_varname: Variable name of EEZ grid in grid file
            eez_export: Whether to add EEZs to particle file
            grp_filter: List of groups to release particles from
            grp_var_name: Variable name of group grid in grid file
            grp_export: Whether to add groups to particle file
            coral_varname: Variable name of coral cover grid in grid file
            export_coral: Whether to add coral cover to particle file
            plot: Whether to create a plot of initial particle positions
            plot_colour: Whether to colour particles by \'grp\' or \'eez\'
            plot_fh: File handle for plot file

        """

        if not self.status['grid']:
            raise Exception('Must import grid before particles can be generated!')
        elif 'num_per_cell' not in kwargs.keys():
            raise KeyError('Please make sure \'num_per_cell\' has been passed as an argument')
        elif 'coral_varname' not in kwargs.keys():
            raise KeyError('Must pass coral_varname as an argument')

        if 'export_grp' not in kwargs.keys():
            kwargs['export_grp'] = False

        if 'export_eez' not in kwargs.keys():
            kwargs['export_eez'] = False

        if 'export_coral_cover' not in kwargs.keys():
            kwargs['export_coral_cover'] = False

        if 'eez_filter' not in kwargs.keys():
            kwargs['eez_filter'] = False

        if 'grp_filter' not in kwargs.keys():
            kwargs['grp_filter'] = False

        # Import EEZ/Group grids if necessary
        if kwargs['eez_filter'] or kwargs['export_eez']:
            if 'eez_varname' not in kwargs.keys():
                raise KeyError('Please give the EEZ variable name if you want to filter by EEZ')
            with Dataset(self.fh['grid'], mode='r') as nc:
                eez_grid = nc.variables[kwargs['eez_varname']][:]

            if self.params['grid_type'] == 'A':
                # Make sure dimensions agree
                assert np.array_equiv(np.shape(eez_grid), (self.axes['psi']['ny'],
                                                           self.axes['psi']['nx']))
            else:
                raise NotImplementedError('C-grids have not been implemented yet!')

        if kwargs['grp_filter'] or kwargs['export_grp']:
            if 'grp_varname' not in kwargs.keys():
                raise KeyError('Please give the group variable name if you want to filter by group')
            with Dataset(self.fh['grid'], mode='r') as nc:
                grp_grid = nc.variables[kwargs['grp_varname']][:]
                if np.max(grp_grid) > np.iinfo(np.uint8).max:
                    raise NotImplementedError('Maximum group number currently limited to 255')

            if self.params['grid_type'] == 'A':
                # Make sure dimensions agree
                assert np.array_equiv(np.shape(grp_grid), (self.axes['psi']['ny'],
                                                           self.axes['psi']['nx']))
            else:
                raise NotImplementedError('C-grids have not been implemented yet!')

        # Import coral mask
        with Dataset(self.fh['grid'], mode='r') as nc:
            coral_grid = nc.variables[kwargs['coral_varname']][:]

        if self.params['grid_type'] == 'A':
            # Make sure dimensions agree
            assert np.array_equiv(np.shape(coral_grid), (self.axes['psi']['ny'],
                                                         self.axes['psi']['nx']))
        else:
            raise NotImplementedError('C-grids have not been implemented yet!')

        # Build a mask of valid locations
        coral_mask = (coral_grid > 0)
        if kwargs['eez_filter']:
            coral_mask *= np.isin(eez_grid, kwargs['eez_filter'])
        if kwargs['grp_filter']:
            coral_mask *= np.isin(grp_grid, kwargs['grp_filter'])

        nl = int(np.sum(coral_mask)) # Number of sites identified

        if nl == 0:
            raise Exception('No valid coral sites found')
        else:
            print(str(nl) + ' coral sites identified.')
            print()

        coral_yidx, coral_xidx = np.where(coral_mask)

        # Calculate the number of particles to release per cell (if not square)
        self.params['pn'] = int(np.ceil(kwargs['num_per_cell']**0.5))
        self.params['pn2'] = int(self.params['pn']**2)

        if np.ceil(kwargs['num_per_cell']**0.5) != kwargs['num_per_cell']**0.5:
            print('Particle number per cell is not square.')
            print('Old particle number: ' + str(kwargs['num_per_cell']))
            print('New particle number: ' + str(int(self.params['pn']**2)))
            print()

        # Build a list of initial particle locations
        lon_rho_grid, lat_rho_grid = np.meshgrid(self.axes['rho']['lon'],
                                                 self.axes['rho']['lat'])
        lon_psi_grid, lat_psi_grid = np.meshgrid(self.axes['psi']['lon'],
                                                 self.axes['psi']['lat'])

        particles = {} # Dictionary to hold initial particle properties
        particles['lon'] = np.zeros((nl*self.params['pn2'],), dtype=np.float64)
        particles['lat'] = np.zeros((nl*self.params['pn2'],), dtype=np.float64)

        print(str(len(particles['lon'])) + ' particles generated.')

        if kwargs['export_coral_cover']:
            if np.max(coral_grid) < np.iinfo(np.int32).max:
                particles['coral_cover'] = np.zeros((nl*self.params['pn2'],),
                                                    dtype=np.int32)
            else:
                particles['coral_cover'] = np.zeros((nl*self.params['pn2'],),
                                                    dtype=np.int64)
        if kwargs['export_eez']:
            if np.max(eez_grid) < np.iinfo(np.uint8).max:
                particles['eez'] = np.zeros((nl*self.params['pn2'],),
                                            dtype=np.uint8)
            else:
                particles['eez'] = np.zeros((nl*self.params['pn2'],),
                                             dtype=np.uint16)

        if kwargs['export_grp']:
            if np.max(grp_grid) < np.iinfo(np.uint8).max:
                particles['grp'] = np.zeros((nl*self.params['pn2'],),
                                            dtype=np.uint8)
            else:
                particles['grp'] = np.zeros((nl*self.params['pn2'],),
                                            dtype=np.uint16)

        if self.params['grid_type'] == 'A':
            # For cell psi[i, j], the surrounding rho cells are:
            # rho[i, j]     (SW)
            # rho[i, j+1]   (SE)
            # rho[i+1, j]   (NW)
            # rho[i+1, j+1] (NE)

            for k, (i, j) in enumerate(zip(coral_yidx, coral_xidx)):
                # Firstly calculate the basic particle grid (may be variable for
                # curvilinear grids)

                dX = lon_rho_grid[i, j+1] - lon_rho_grid[i, j] # Grid spacing
                dY = lat_rho_grid[i+1, j] - lat_rho_grid[i, j] # Grid spacing
                dx = dX/self.params['pn']                      # Particle spacing
                dy = dY/self.params['pn']                      # Particle spacing

                gx = np.linspace(lon_rho_grid[i, j]+(dx/2),    # Particle x locations
                                 lon_rho_grid[i, j+1]-(dx/2), num=self.params['pn'])

                gy = np.linspace(lat_rho_grid[i, j]+(dy/2),    # Particle y locations
                                 lat_rho_grid[i+1, j]-(dy/2), num=self.params['pn'])

                gx, gy = [grid.flatten() for grid in np.meshgrid(gx, gy)] # Flattened arrays

                particles['lon'][k*self.params['pn2']:(k+1)*self.params['pn2']] = gx
                particles['lat'][k*self.params['pn2']:(k+1)*self.params['pn2']] = gy

                if kwargs['export_coral_cover']:
                    coral_cover_k = coral_grid[i, j]
                    particles['coral_cover'][k*self.params['pn2']:(k+1)*self.params['pn2']] = coral_cover_k

                if kwargs['export_eez']:
                    eez_k = eez_grid[i, j]
                    particles['eez'][k*self.params['pn2']:(k+1)*self.params['pn2']] = eez_k

                if kwargs['export_grp']:
                    grp_k = grp_grid[i, j]
                    particles['grp'][k*self.params['pn2']:(k+1)*self.params['pn2']] = grp_k

        else:
            raise NotImplementedError('C-grids have not been implemented yet!')

        # Now export to DataFrame
        particles_df = pd.DataFrame({'lon': particles['lon'],
                                     'lat': particles['lat']})

        if kwargs['export_coral_cover']:
            particles_df['coral_cover'] = particles['coral_cover']

        if kwargs['export_eez']:
            particles_df['eez'] = particles['eez']

        if kwargs['export_grp']:
            particles_df['grp'] = particles['grp']

        self.particles = particles_df

        # Now plot (if wished)
        kwargs['plot'] = False if 'plot' not in kwargs.keys() else kwargs['plot']

        if kwargs['plot']:
            # Create a plot of initial particle positions
            kwargs['plot_colour'] = None if 'plot_colour' not in kwargs.keys() else kwargs['plot_colour']
            if kwargs['plot_colour'] == 'grp':
                if kwargs['export_grp']:
                    colour_series = particles['grp']
                else:
                    raise Exception('Cannot plot by group if group is not exported ')
            elif kwargs['plot_colour'] == 'eez':
                if kwargs['export_eez']:
                    colour_series = particles['eez']
                else:
                    raise Exception('Cannot plot by group if EEZ is not exported ')
            else:
                colour_series = 'k'

            plot_x_range = np.max(particles['lon']) - np.min(particles['lon'])
            plot_y_range = np.max(particles['lat']) - np.min(particles['lat'])
            plot_x_range = [np.min(particles['lon']) - 0.1*plot_x_range,
                            np.max(particles['lon']) + 0.1*plot_x_range]
            plot_y_range = [np.min(particles['lat']) - 0.1*plot_y_range,
                            np.max(particles['lat']) + 0.1*plot_y_range]
            aspect = (plot_y_range[1] - plot_y_range[0])/(plot_x_range[1] - plot_x_range[0])

            f, ax = plt.subplots(1, 1, figsize=(20, 20*aspect), subplot_kw={'projection': ccrs.PlateCarree()})
            cmap = 'prism'

            ax.set_xlim(plot_x_range)
            ax.set_ylim(plot_y_range)
            ax.set_title('Initial positions for particles')

            ax.scatter(particles['lon'], particles['lat'], c=colour_series,
                       cmap=cmap, s=1, transform=ccrs.PlateCarree())

            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.8, color='black', linestyle='-', zorder=11)
            gl.ylocator = mticker.FixedLocator(np.arange(-30, 30, 5))
            gl.xlocator = mticker.FixedLocator(np.arange(0, 90, 5))
            gl.xlabels_top = False
            gl.ylabels_right = False

            if 'plot_fh' in kwargs.keys():
                plot_fh = kwargs['plot_fh']
            else:
                warnings.warn('No plotting fh given, saving to script directory')
                plot_fh = self.root_dir + 'initial_particle_positions.png'

            plt.savefig(plot_fh, dpi=300)
            plt.close()

        self.status['release_loc'] = True

    def add_release_time(self, time):
        """
        Parameters (* are required)
        ----------
        kwargs :
            time* : Release time for particles (datetime)

        """
        if not self.status['release_loc']:
            raise Exception('Must generate particles first before times can be assigned')
        elif type(time) != datetime:
            raise Exception('Input time must be a python datetime object')

        if self.status['currents']:
            model_start = pd.Timestamp(self.fieldset.time_origin.time_origin)
            particle_start = pd.Timestamp(time)

            if particle_start < model_start:
                warnings.warn(('Particles have been initialised at ' +
                               str(particle_start) + ' but model data starts at ' +
                               str(model_start) + '. Shifting particle start to' +
                               ' model start'))
                time = model_start
        else:
            warnings.warn(('Currents have not been imported yet: cannot check '
                           'whether particle release time is within simulation '
                           'timespan.'))

        self.particles['release_time'] = time

        self.status['release_time'] = True

    def add_fields(self, fields):
        """
        Parameters (* are required)
        ----------
        kwargs :
            fields* : Dictionary of TRACER fields in grid file to add to fieldset
                      At present, the following keys are required:
                          'groups': Coral groups
                          'coral_cover': Coral cover

        """

        if not self.status['currents']:
            raise Exception('Must generate fieldset first with \'import_currents\'')

        if not all (key in fields.keys() for key in ['groups', 'coral_fraction']):
            raise KeyError('Please make sure all required fields are given')
        else:
            self.params['groups_varname'] = fields['groups']
            self.params['coral_fraction_varname'] = fields['coral_fraction']

        if self.params['groups_varname'] != 'coral_grp_c' or self.params['coral_fraction_varname'] != 'coral_frac_c':
            raise NotImplementedError('Groups variable must currently be coral_grp_c and coral fraction variable coral_frac_c')

        if self.params['grid_type'] == 'A':
            for field in fields.values():
                # Firstly check that the field has the correct dimensions (i.e. PSI for A-Grid)
                with Dataset(self.fh['grid'], mode='r') as nc:
                    temp_field = nc.variables[field][:]
                    if not np.array_equiv(np.shape(temp_field),
                                          (self.axes['psi']['ny'], self.axes['psi']['nx'])):
                        raise Exception('Field ' + str(field) + ' has incorrect dimensions')

                if 'psi' not in self.params['dimension_names']:
                    raise NotImplementedError('Psi grid must currently still be supplied in netcdf file')

                temp_field = Field.from_netcdf(self.fh['grid'],
                                               variable=field,
                                               dimensions=self.params['dimension_names']['psi'],
                                               interp_method='nearest',
                                               allow_time_extrapolation=True)
                self.fieldset.add_field(temp_field)
        else:
            raise NotImplementedError('C-grids have not been implemented yet!')

    def create_particleset(self, **kwargs):
        """
        Parameters (* are required)
        ----------
        kwargs :
            test: Whether to run the testing kernels (bool)
            fh*: File handle for output netcdf

        """

        if 'test' not in kwargs:
            kwargs['test'] = False

        if 'fh' not in kwargs:
            raise KeyError('Filehandle required for output')
        else:
            self.fh['traj'] = kwargs['fh']

        if not kwargs['test']:
            class larva(JITParticle):

                ###################################################################
                # TEMPORARY VARIABLES FOR TRACKING PARTICLE POSITION/STATUS       #
                ###################################################################

                # Group of current cell (>0 if in any reef cell)
                grp = Variable('grp',
                               dtype=np.int8,
                               initial=0,
                               to_write=False)

                # Reef fraction of current cell (>0 if in any reef cell)
                rf = Variable('rf',
                              dtype=np.float32,
                              initial=0,
                              to_write=False)

                # Time at sea (Total time since spawning)
                ot  = Variable('ot',
                               dtype=np.int32,
                               initial=0,
                               to_write=False)

                # Phi (int_0^t F(tau) dtau, where F(tau) is the coral cell fraction
                #      at time tau)
                phi = Variable('phi',
                               dtype=np.float32,
                               initial=0,
                               to_write=False)

                ##########################################################################
                # PROVENANCE IDENTIFIERS #################################################
                ##########################################################################

                # Group of parent reef
                grp0 = Variable('grp0',
                                dtype=np.int8,
                                to_write=True)

                # Original longitude
                lon0 = Variable('lon0',
                                dtype=np.float32,
                                to_write=True)

                # Original latitude
                lat0 = Variable('lat0',
                                dtype=np.float32,
                                to_write=True)

                # Reef area of parent reef
                ra0 = Variable('ra0',
                               dtype=np.int32,
                               to_write=True)

                ##########################################################################
                # TEMPORARY VARIABLES FOR TRACKING BEACHING AT SPECIFIED REEF SITES ######
                ##########################################################################

                # Current reef time (memory of time in CURRENT reef group - in time-steps)
                current_reef_time = Variable('current_reef_time',
                                             dtype=np.uint16,
                                             initial=0,
                                             to_write=False)

                # Current reef t0 (memory of time when the CURRENT reef group was reached - in time-steps)
                current_reef_t0 = Variable('current_reef_t0',
                                           dtype=np.uint16,
                                           initial=0,
                                           to_write=False)

                # Current reef Phi (memory of accumulatd phi inthe CURRENT reef group)
                current_reef_phi = Variable('current_reef_phi',
                                             dtype=np.float32,
                                             initial=0.,
                                             to_write=False)

                # Current reef phi0 (memory of phi when the CURRENT reef group was reached)
                current_reef_phi0 = Variable('current_reef_phi0',
                                             dtype=np.float32,
                                             initial=0.,
                                             to_write=False)

                # Current reef group (memory of the CURRENT reef group)
                current_reef_grp = Variable('current_reef_grp',
                                            dtype=np.uint8,
                                            initial=0.,
                                            to_write=False)

                ##########################################################################
                # RECORD OF ALL EVENTS ###################################################
                ##########################################################################

                # Number of events
                e_num = Variable('e_num', dtype=np.int16, initial=0, to_write=True)

                # Event variables (g = group, t = time in group, s = t0, p = phi0, f = phi of group)
                g0 = Variable('g0', dtype=np.uint8, initial=0, to_write=True)
                t0 = Variable('t0', dtype=np.uint16, initial=0, to_write=True)
                s0 = Variable('t0', dtype=np.uint16, initial=0, to_write=True)
                p0 = Variable('p0', dtype=np.float32, initial=0., to_write=True)
                f0 = Variable('f0', dtype=np.float32, initial=0., to_write=True)

                g1 = Variable('g1', dtype=np.uint8, initial=0, to_write=True)
                t1 = Variable('t1', dtype=np.uint16, initial=0, to_write=True)
                s1 = Variable('t1', dtype=np.uint16, initial=0, to_write=True)
                p1 = Variable('p1', dtype=np.float32, initial=0., to_write=True)
                f1 = Variable('f1', dtype=np.float32, initial=0., to_write=True)

                g2 = Variable('g2', dtype=np.uint8, initial=0, to_write=True)
                t2 = Variable('t2', dtype=np.uint16, initial=0, to_write=True)
                s2 = Variable('t2', dtype=np.uint16, initial=0, to_write=True)
                p2 = Variable('p2', dtype=np.float32, initial=0., to_write=True)
                f2 = Variable('f2', dtype=np.float32, initial=0., to_write=True)

                g3 = Variable('g3', dtype=np.uint8, initial=0, to_write=True)
                t3 = Variable('t3', dtype=np.uint16, initial=0, to_write=True)
                s3 = Variable('t3', dtype=np.uint16, initial=0, to_write=True)
                p3 = Variable('p3', dtype=np.float32, initial=0., to_write=True)
                f3 = Variable('f3', dtype=np.float32, initial=0., to_write=True)

                g4 = Variable('g4', dtype=np.uint8, initial=0, to_write=True)
                t4 = Variable('t4', dtype=np.uint16, initial=0, to_write=True)
                s4 = Variable('t4', dtype=np.uint16, initial=0, to_write=True)
                p4 = Variable('p4', dtype=np.float32, initial=0., to_write=True)
                f4 = Variable('f4', dtype=np.float32, initial=0., to_write=True)

        else:
            class larva(JITParticle):

                ###################################################################
                # TEMPORARY VARIABLES FOR TRACKING PARTICLE POSITION/STATUS       #
                ###################################################################

                # Group of current cell (>0 if in any reef cell)
                grp = Variable('grp',
                               dtype=np.int8,
                               initial=0,
                               to_write=False)

                # Reef fraction of current cell (>0 if in any reef cell)
                rf = Variable('rf',
                              dtype=np.float32,
                              initial=0,
                              to_write=False)

                # Time at sea (Total time since spawning)
                ot  = Variable('ot',
                               dtype=np.int32,
                               initial=0,
                               to_write=False)

                # Phi (int_0^t F(tau) dtau, where F(tau) is the coral cell fraction
                #      at time tau)
                phi = Variable('phi',
                               dtype=np.float32,
                               initial=0,
                               to_write=False)

                ##########################################################################
                # PROVENANCE IDENTIFIERS #################################################
                ##########################################################################

                # Group of parent reef
                grp0 = Variable('grp0',
                                dtype=np.int8,
                                to_write=True)

                # Original longitude
                lon0 = Variable('lon0',
                                dtype=np.float32,
                                to_write=True)

                # Original latitude
                lat0 = Variable('lat0',
                                dtype=np.float32,
                                to_write=True)

                # Reef area of parent reef
                ra0 = Variable('ra0',
                               dtype=np.int32,
                               to_write=True)

                ##########################################################################
                # TEMPORARY VARIABLES FOR TRACKING BEACHING AT SPECIFIED REEF SITES ######
                ##########################################################################

                # Current reef time (memory of time in CURRENT reef group - in time-steps)
                current_reef_time = Variable('current_reef_time',
                                             dtype=np.uint16,
                                             initial=0,
                                             to_write=False)

                # Current reef t0 (memory of time when the CURRENT reef group was reached - in time-steps)
                current_reef_t0 = Variable('current_reef_t0',
                                           dtype=np.uint16,
                                           initial=0,
                                           to_write=False)

                # Current reef Phi (memory of accumulatd phi inthe CURRENT reef group)
                current_reef_phi = Variable('current_reef_phi',
                                             dtype=np.float32,
                                             initial=0.,
                                             to_write=False)

                # Current reef phi0 (memory of phi when the CURRENT reef group was reached)
                current_reef_phi0 = Variable('current_reef_phi0',
                                             dtype=np.float32,
                                             initial=0.,
                                             to_write=False)

                # Current reef group (memory of the CURRENT reef group)
                current_reef_grp = Variable('current_reef_grp',
                                            dtype=np.uint8,
                                            initial=0.,
                                            to_write=False)

                ##########################################################################
                # RECORD OF ALL EVENTS ###################################################
                ##########################################################################

                # Number of events
                e_num = Variable('e_num', dtype=np.int16, initial=0, to_write=True)

                # Event variables (g = group, t = time in group, s = t0, p = phi0, f = phi of group)
                g0 = Variable('g0', dtype=np.uint8, initial=0, to_write=True)
                t0 = Variable('t0', dtype=np.uint16, initial=0, to_write=True)
                s0 = Variable('s0', dtype=np.uint16, initial=0, to_write=True)
                p0 = Variable('p0', dtype=np.float32, initial=0., to_write=True)
                f0 = Variable('f0', dtype=np.float32, initial=0., to_write=True)

                g1 = Variable('g1', dtype=np.uint8, initial=0, to_write=True)
                t1 = Variable('t1', dtype=np.uint16, initial=0, to_write=True)
                s1 = Variable('s1', dtype=np.uint16, initial=0, to_write=True)
                p1 = Variable('p1', dtype=np.float32, initial=0., to_write=True)
                f1 = Variable('f1', dtype=np.float32, initial=0., to_write=True)

                g2 = Variable('g2', dtype=np.uint8, initial=0, to_write=True)
                t2 = Variable('t2', dtype=np.uint16, initial=0, to_write=True)
                s2 = Variable('s2', dtype=np.uint16, initial=0, to_write=True)
                p2 = Variable('p2', dtype=np.float32, initial=0., to_write=True)
                f2 = Variable('f2', dtype=np.float32, initial=0., to_write=True)

                g3 = Variable('g3', dtype=np.uint8, initial=0, to_write=True)
                t3 = Variable('t3', dtype=np.uint16, initial=0, to_write=True)
                s3 = Variable('s3', dtype=np.uint16, initial=0, to_write=True)
                p3 = Variable('p3', dtype=np.float32, initial=0., to_write=True)
                f3 = Variable('f3', dtype=np.float32, initial=0., to_write=True)

                g4 = Variable('g4', dtype=np.uint8, initial=0, to_write=True)
                t4 = Variable('t4', dtype=np.uint16, initial=0, to_write=True)
                s4 = Variable('s4', dtype=np.uint16, initial=0, to_write=True)
                p4 = Variable('p4', dtype=np.float32, initial=0., to_write=True)
                f4 = Variable('f4', dtype=np.float32, initial=0., to_write=True)

                larval_loss0 = Variable('larval_loss0', dtype=np.float32, initial=0., to_write=True)
                larval_loss1 = Variable('larval_loss1', dtype=np.float32, initial=0., to_write=True)
                larval_loss2 = Variable('larval_loss2', dtype=np.float32, initial=0., to_write=True)
                larval_loss3 = Variable('larval_loss3', dtype=np.float32, initial=0., to_write=True)
                larval_loss4 = Variable('larval_loss4', dtype=np.float32, initial=0., to_write=True)

                larval_number = Variable('larval_number', dtype=np.float32, initial=0., to_write=False)
                current_reef_larval_number0 = Variable('current_reef_larval_number0', dtype=np.float32, initial=0., to_write=False)

        self.pset = ParticleSet.from_list(fieldset=self.fieldset,
                                          pclass=larva,
                                          lonlatdepth_dtype=np.float64,
                                          lon=self.particles['lon'],
                                          lat=self.particles['lat'],
                                          time=self.particles['release_time'],
                                          lon0=self.particles['lon'],
                                          lat0=self.particles['lat'],
                                          grp0=self.particles['grp'],
                                          ra0=self.particles['coral_cover'])

        if not kwargs['test']:
            self.trajectory_file = self.pset.ParticleFile(name=self.fh['traj'], write_ondelete=True)
        else:
            self.trajectory_file = self.pset.ParticleFile(name=self.fh['traj'], outputdt=timedelta(hours=1))


    def create_kernels(self, **kwargs):
        """
        Parameters (* are required)
        ----------
        kwargs :
            competency_period: Period until which larvae cannot settle (timedelta)
            diffusion: Either False for no diffusion, or the horizontal diffusivity (m^2/s)
            dt*: Parcels RK4 timestep (timedelta)
            run_time*: Total runtime (timedelta)
            test: Whether to run the testing kernels (bool)

        """

        if 'competency_period' in kwargs:
            self.params['competency_period'] = kwargs['competency_period'].total_seconds()
            if self.params['competency_period'] > 0:
                raise NotImplementedError('Competency not yet implemented')
        else:
            warnings.warn('No competency period set')
            self.params['competency_period'] = 0

        if 'diffusion' in kwargs:
            self.params['Kh'] = kwargs['diffusion']
            if kwargs['diffusion']:
                raise NotImplementedError('Diffusion not yet implemented')
        else:
            self.params['Kh'] = None

        if not all (key in kwargs.keys() for key in ['dt', 'run_time']):
            raise KeyError('Must supply Parcels RK4 timestep and run time for kernel creation')

        if 'test' not in kwargs:
            kwargs['test'] = False

        assert kwargs['run_time'].total_seconds()/kwargs['dt'].total_seconds() < np.iinfo(np.uint16).max

        self.fieldset.add_constant('max_age', kwargs['run_time'].total_seconds() - kwargs['dt'].total_seconds())
        self.params['run_time'] = kwargs['run_time']
        self.params['dt'] = kwargs['dt']

        # Create Particle Class
        if not kwargs['test']:
            # Controller for managing particle events
            def event(particle, fieldset, time):

                # 1 Keep track of the amount of time spent at sea
                particle.ot += particle.dt

                # 2 Assess reef status
                particle.grp = fieldset.coral_grp_c[particle]

                save_event = False
                new_event = False

                # 3 Trigger event cascade if larva is in a reef site
                if particle.grp > 0:

                    particle.rf = fieldset.coral_frac_c[particle]

                    # Check if an event has already been triggered
                    if particle.current_reef_time > 0:

                        # Check if we are in the same reef group as the current event
                        if particle.grp == particle.current_reef_grp:

                            # If contiguous event, just add time and phi
                            particle.current_reef_time += 1
                            particle.phi += particle.rf*particle.dt

                            # But also check that the particle isn't about to expire (save if so)
                            # Otherwise particles hanging around reefs at the end of the simulation
                            # won't get saved.

                            if particle.ot > fieldset.max_age:
                                save_event = True

                        else:

                            # Otherwise, we need to save the old event and create a new event
                            save_event = True
                            new_event = True

                    else:

                        # If event has not been triggered, create a new event
                        new_event = True

                else:

                    # Otherwise, check if ongoing event has just ended
                    if particle.current_reef_time > 0:

                        save_event = True

                if save_event:
                    # Save current values
                    # Unfortunately, due to the limited functions allowed in parcels, this
                    # required an horrendous if-else chain

                    if particle.e_num == 0:
                        particle.g0 = particle.current_reef_grp
                        particle.t0 = particle.current_reef_time
                        particle.s0 = particle.current_reef_t0
                        particle.p0 = particle.current_reef_phi0
                        particle.f0 = particle.current_reef_phi
                    elif particle.e_num == 1:
                        particle.g1 = particle.current_reef_grp
                        particle.t1 = particle.current_reef_time
                        particle.s1 = particle.current_reef_t0
                        particle.p1 = particle.current_reef_phi0
                        particle.f1 = particle.current_reef_phi
                    elif particle.e_num == 2:
                        particle.g2 = particle.current_reef_grp
                        particle.t2 = particle.current_reef_time
                        particle.s2 = particle.current_reef_t0
                        particle.p2 = particle.current_reef_phi0
                        particle.f2 = particle.current_reef_phi
                    elif particle.e_num == 3:
                        particle.g3 = particle.current_reef_grp
                        particle.t3 = particle.current_reef_time
                        particle.s3 = particle.current_reef_t0
                        particle.p3 = particle.current_reef_phi0
                        particle.f3 = particle.current_reef_phi
                    elif particle.e_num == 4:
                        particle.g4 = particle.current_reef_grp
                        particle.t4 = particle.current_reef_time
                        particle.s4 = particle.current_reef_t0
                        particle.p4 = particle.current_reef_phi0
                        particle.f4 = particle.current_reef_phi

                        particle.delete() # Delete particle, since no more reefs can be saved

                    # Then reset current values to zero
                    particle.current_reef_grp = 0
                    particle.current_reef_time = 0
                    particle.current_reef_t0 = 0
                    particle.current_reef_phi0 = 0
                    particle.current_reef_phi = 0

                    # Add to event number counter
                    particle.e_num += 1

                if new_event:
                    # Add status to current (for current event) values
                    # Timesteps at current reef
                    particle.current_reef_time = 1

                    # Timesteps spent in the ocean overall upon arrival (minus one, before this step)
                    particle.current_reef_t0 = (particle.ot/particle.dt) - 1

                    # Phi upon arrival
                    particle.current_reef_phi0 = particle.phi

                    # Phi at current reef
                    particle.current_reef_phi += particle.rf*particle.dt

                    # Current reef group
                    particle.current_reef_grp = particle.grp

                # Finally, check if particle needs to be deleted
                if particle.ot >= fieldset.max_age:

                    # Only delete particles where at least 1 event has been recorded
                    if particle.e_num > 0:
                        particle.delete()
        else:
            # Test event controller for validation
            def event(particle, fieldset, time):
                lm = 8e-7
                ls = 1e-5

                # 1 Keep track of the amount of time spent at sea
                particle.ot += particle.dt

                # 2 Assess reef status
                particle.grp = fieldset.coral_grp_c[particle]

                save_event = False
                new_event = False

                # 3 Trigger event cascade if larva is in a reef site
                if particle.grp > 0:

                    particle.rf = fieldset.coral_frac_c[particle]

                    # Check if an event has already been triggered
                    if particle.current_reef_time > 0:

                        # Check if we are in the same reef group as the current event
                        if particle.grp == particle.current_reef_grp:

                            # If contiguous event, just add time and phi
                            particle.current_reef_time += 1
                            particle.phi += particle.rf*particle.dt
                            particle.larval_number = particle.larval_number - (particle.larval_number)*(lm + (ls*particle.rf))*(particle.dt)

                            # But also check that the particle isn't about to expire (save if so)
                            # Otherwise particles hanging around reefs at the end of the simulation
                            # won't get saved.

                            if particle.ot > fieldset.max_age:
                                save_event = True

                        else:

                            # Otherwise, we need to save the old event and create a new event
                            save_event = True
                            new_event = True

                    else:

                        # If event has not been triggered, create a new event
                        new_event = True

                else:

                    # Otherwise, check if ongoing event has just ended
                    if particle.current_reef_time > 0:

                        save_event = True

                if save_event:
                    # Save current values
                    # Unfortunately, due to the limited functions allowed in parcels, this
                    # required an horrendous if-else chain

                    if particle.e_num == 0:
                        particle.g0 = particle.current_reef_grp
                        particle.t0 = particle.current_reef_time
                        particle.s0 = particle.current_reef_t0
                        particle.p0 = particle.current_reef_phi0
                        particle.f0 = particle.current_reef_phi
                        particle.larval_loss0 = particle.current_reef_larval_number0 - particle.larval_number
                    elif particle.e_num == 1:
                        particle.g1 = particle.current_reef_grp
                        particle.t1 = particle.current_reef_time
                        particle.s1 = particle.current_reef_t0
                        particle.p1 = particle.current_reef_phi0
                        particle.f1 = particle.current_reef_phi
                        particle.larval_loss1 = particle.current_reef_larval_number0 - particle.larval_number
                    elif particle.e_num == 2:
                        particle.g2 = particle.current_reef_grp
                        particle.t2 = particle.current_reef_time
                        particle.s2 = particle.current_reef_t0
                        particle.p2 = particle.current_reef_phi0
                        particle.f2 = particle.current_reef_phi
                        particle.larval_loss2 = particle.current_reef_larval_number0 - particle.larval_number
                    elif particle.e_num == 3:
                        particle.g3 = particle.current_reef_grp
                        particle.t3 = particle.current_reef_time
                        particle.s3 = particle.current_reef_t0
                        particle.p3 = particle.current_reef_phi0
                        particle.f3 = particle.current_reef_phi
                        particle.larval_loss3 = particle.current_reef_larval_number0 - particle.larval_number
                    elif particle.e_num == 4:
                        particle.g4 = particle.current_reef_grp
                        particle.t4 = particle.current_reef_time
                        particle.s4 = particle.current_reef_t0
                        particle.p4 = particle.current_reef_phi0
                        particle.f4 = particle.current_reef_phi
                        particle.larval_loss4 = particle.current_reef_larval_number0 - particle.larval_number

                        particle.delete() # Delete particle, since no more reefs can be saved

                    # Then reset current values to zero
                    particle.current_reef_grp = 0
                    particle.current_reef_time = 0
                    particle.current_reef_t0 = 0
                    particle.current_reef_phi0 = 0
                    particle.current_reef_phi = 0

                    # Add to event number counter
                    particle.e_num += 1

                if new_event:
                    # Add status to current (for current event) values
                    # Timesteps at current reef
                    particle.current_reef_time = 1

                    # Timesteps spent in the ocean overall upon arrival (minus one, before this step)
                    particle.current_reef_t0 = (particle.ot/particle.dt) - 1

                    # Phi upon arrival
                    particle.current_reef_phi0 = particle.phi

                    # Phi at current reef
                    particle.current_reef_phi += particle.rf*particle.dt

                    # Current reef group
                    particle.current_reef_grp = particle.grp

                    # Larval number at start of current group
                    particle.current_reef_larval_number0 = particle.larval_number

                    particle.larval_number = particle.larval_number - (particle.larval_number)*(lm + (ls*particle.rf))*(particle.dt)

                # Finally, check if particle needs to be deleted
                if particle.ot >= fieldset.max_age:

                    # Only delete particles where at least 1 event has been recorded
                    if particle.e_num > 0:
                        particle.delete()

        self.kernels = (self.pset.Kernel(AdvectionRK4) +
                        self.pset.Kernel(event))

    def run(self):
        """
        This function runs OceanParcels

        """

        def deleteParticle(particle, fieldset, time):
            #  Recovery kernel to delete a particle if an error occurs
            particle.delete()

        self.pset.execute(self.kernels,
                          runtime=self.params['run_time'],
                          dt=self.params['dt'],
                          recovery={ErrorCode.ErrorOutOfBounds: deleteParticle,
                                    ErrorCode.ErrorInterpolation: deleteParticle},
                          output_file=self.trajectory_file)


        self.trajectory_file.export()

























