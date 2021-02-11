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


def bins_map(bins, bins_list, trips):
    # Plot all the bin trips on the map, use different colors for different bins.
    # Each color represents one bin based on index
    # Choose the first start point from the trips to locate the map for convenience

    # The function takes three parameters:
    # - bins: bins list above cutoff after running pipeline.remove_noise
    # - bins_list: the list of indices of selected bins
    # - trips: the trips after running pipeline.read_data

    # In order to plot trips correctly, map index should be len(bins)+1,
    # so that the color steps length is equal to the bins number. For bin index which is 0,
    # will be plotted in color which is in index 1.
    color_map = cm.linear.Set1_07.to_step(len(bins)+1, index=[i for i in range(len(bins)+1)])
    map = folium.Map(location=[trips[0].data.start_loc["coordinates"][1], trips[0].data.start_loc["coordinates"][0]],
                   zoom_start=12, max_zoom=30, control_scale=True)

    if bins_list:
        for bin_index in bins_list:
            curr_bin = bins[bins_list[bin_index]]
            map_index = bins_list[bin_index]+1
            for curr_trip_index in curr_bin:
                curr_trip = trips[curr_trip_index]
                # We need polyline to plot the trip according to start_loc and end_loc
                # Flip indices because points are in geojson (i.e. lon, lat),folium takes [lat,lon]
                layer = folium.PolyLine(
                    [[curr_trip.data.start_loc["coordinates"][1], curr_trip.data.start_loc["coordinates"][0]],
                     [curr_trip.data.end_loc["coordinates"][1], curr_trip.data.end_loc["coordinates"][0]]], weight=2,
                    color=color_map(map_index))
                layer.add_to(map)

    else:
        for index, curr_bin in enumerate(bins):
            for curr_trip_index in curr_bin:
                curr_trip = trips[curr_trip_index]
                layer = folium.PolyLine(
                    [[curr_trip.data.start_loc["coordinates"][1], curr_trip.data.start_loc["coordinates"][0]],
                     [curr_trip.data.end_loc["coordinates"][1], curr_trip.data.end_loc["coordinates"][0]]], weight=2,
                    color=color_map(index))
                layer.add_to(map)
    map.add_child(color_map)
    return map


def clusters_map(labels,labels_list, points, clusters):
    # Each color represents a label
    # labels have to be in order in the colormap index

    # Plot all clusters with different colors on the map
    # Choose the first start point to locate the map

    # The function takes four parameters:
    # - labels: trip label list after running featurization/pipeline.cluster
    # - labels_list: the list of selected label
    # - points: coordinate points from featurization/pipeline.cluster
    # - clusters: number of clusters
    map = folium.Map(location=[points[0][1], points[0][0]], zoom_start=12, max_zoom=30, control_scale=True)
    labels_clt = list(set(sorted(labels)))
    color_map = cm.linear.Set1_07.to_step(clusters + 1, index=[i for i in range(len(labels_clt) + 1)])

    if labels_list:
        for label in labels_list:
            for i, point in enumerate(points):
                if labels[i] == label:
                    start_lat = point[1]
                    start_lon = point[0]
                    end_lat = point[3]
                    end_lon = point[2]
                    layer = folium.PolyLine([[start_lat, start_lon],
                                             [end_lat, end_lon]], weight=2, color=color_map(labels[i]+1))
                    layer.add_to(map)
    else:
        if labels:
            for i, point in enumerate(points):
                start_lat = point[1]
                start_lon = point[0]
                end_lat = point[3]
                end_lon = point[2]
                layer = folium.PolyLine([[start_lat, start_lon],
                                         [end_lat, end_lon]], weight=2, color=color_map(labels[i]+1))
                layer.add_to(map)
    map.add_child(color_map)
    return map
