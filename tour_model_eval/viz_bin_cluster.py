import matplotlib
import matplotlib.pyplot as plt
import numpy
import emission.core.get_database as edb
import logging
import math
import emission.storage.timeseries.abstract_timeseries as esta
import emission.analysis.modelling.tour_model.cluster_pipeline as eamtc
import emission.analysis.modelling.tour_model.similarity as similarity
import emission.analysis.modelling.tour_model.cluster_pipeline as pipeline
import emission.analysis.modelling.tour_model.featurization as featurization

# imports for visualization code
import folium
import branca.colormap as cm


def bins_map(bins, trips):
    # Plot all the bin trips on the map, use different colors for different bins.
    # Each color represents one bin based on index
    # Choose the first start point from the trips to locate the map for convenience
    color_map = cm.linear.Set1_07.to_step(len(bins),index=[i for i in range (len(bins))])
    map = folium.Map(location=[trips[0].data.start_loc["coordinates"][1], trips[0].data.start_loc["coordinates"][0]],
                   zoom_start=12, max_zoom=30, control_scale=True)
    for index, curr_bin in enumerate(bins):
        for curr_trip_index in curr_bin:
            curr_trip = trips[curr_trip_index]
            # We need polyline to plot the trip according to start_loc and end_loc
            # Flip indices because points are in geojson (i.e. lon, lat),folium takes [lat,lon]
            layer = folium.PolyLine(
                [[curr_trip.data.start_loc["coordinates"][1], curr_trip.data.start_loc["coordinates"][0]],
                 [curr_trip.data.end_loc["coordinates"][1], curr_trip.data.end_loc["coordinates"][0]]], weight=2,
                color=color_map(index))
            layer.add_to(map)
    map.add_child(color_map)
    return map


# def cutoff_bins_map(bins, trips):
#     color_map = cm.linear.Set1_07.to_step(len(bins), index=[i for i in range(len(bins))])
#     map = folium.Map(
#         location=[trips[0].data.start_loc["coordinates"][1], trips[0].data.start_loc["coordinates"][0]],
#         zoom_start=12, max_zoom=30, control_scale=True)
#     for t, bin in enumerate(bins):
#         for trip_index in bin:
#             # curr_trip = bin_trips[curr_trip_index]
#             curr_trip = trips[trip_index]
#             layer = folium.PolyLine(
#                 [[curr_trip.data.start_loc["coordinates"][1], curr_trip.data.start_loc["coordinates"][0]],
#                  [curr_trip.data.end_loc["coordinates"][1], curr_trip.data.end_loc["coordinates"][0]]], weight=2,
#                 color=color_map(t))
#             layer.add_to(map)
#     map.add_child(color_map)
#     return map


def clusters_map(labels, points, clusters):
    # Each color represents a label
    # labels have to be in order in the colormap index
    labels_clt = list(set(sorted(labels)))
    color_map = cm.linear.Set1_07.to_step(clusters, index=[i for i in labels_clt])

    # Plot all clusters with different colors on the map
    # Choose the first start point to locate the map
    map = folium.Map(location=[points[0][1], points[0][0]], zoom_start=12, max_zoom=30, control_scale=True)

    if labels:
        for i, point in enumerate(points):
            start_lat = point[1]
            start_lon = point[0]
            end_lat = point[3]
            end_lon = point[2]
            layer = folium.PolyLine([[start_lat, start_lon],
                                     [end_lat, end_lon]], weight=2, color=color_map(labels[i]))
            layer.add_to(map)
    map.add_child(color_map)
    return map


def specific_bin_map(bins, trips):
    color_map = cm.linear.Set1_07.to_step(len(bins), index=[i for i in range(len(bins))])
    # Plot trips in the same bin,
    map = folium.Map(
        location=[trips[0].data.start_loc["coordinates"][1], trips[0].data.start_loc["coordinates"][0]],
        zoom_start=12, max_zoom=30, control_scale=True)

    # Here we to choose a specific bin
    for t in range(0, 3):
        for i in range(len(bins[t])):
            curr_trip = trips[bins[t][i]]
            layer = folium.PolyLine(
                [[curr_trip.data.start_loc["coordinates"][1], curr_trip.data.start_loc["coordinates"][0]],
                 [curr_trip.data.end_loc["coordinates"][1], curr_trip.data.end_loc["coordinates"][0]]], weight=2,
                color=color_map(t))

            layer.add_to(map)
    map.add_child(color_map)
    return map


def specific_cluster_map(labels, points, clusters):
    labels_clt = list(set(sorted(labels)))
    color_map = cm.linear.Set1_07.to_step(clusters, index=[i for i in labels_clt])

    # Plot all clusters with different colors on the map
    # Choose the first start point to locate the map
    map = folium.Map(location=[points[0][1], points[0][0]], zoom_start=12, max_zoom=30,
                             control_scale=True)

    if labels:
        for i, point in enumerate(points):
            # Here we can choose a specific cluster to plot
            if labels[i] == 1:
                start_lat = point[1]
                start_lon = point[0]
                end_lat = point[3]
                end_lon = point[2]
                layer = folium.PolyLine([[start_lat, start_lon],
                                         [end_lat, end_lon]], weight=2, color=color_map(labels[i]))
                layer.add_to(map)
    map.add_child(color_map)
    return map
