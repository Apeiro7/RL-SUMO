import csv
import xml.etree.ElementTree as ET

# Load the XML data
tree = ET.parse('my_emission_file.xml')
root = tree.getroot()

# Define CSV file and headers
csv_file = open('output.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['time', 'vehicle_id', 'CO2', 'CO', 'HC', 'NOx', 'PMx', 'fuel', 'electricity', 'noise', 'route', 'type', 'waiting', 'lane', 'pos', 'speed', 'angle', 'x', 'y'])

# Extract data from XML and write to CSV
for timestep in root.findall('.//timestep'):
    for vehicle in timestep.findall('.//vehicle'):
        row = [
            timestep.get('time'),
            vehicle.get('id'),
            vehicle.get('CO2'),
            vehicle.get('CO'),
            vehicle.get('HC'),
            vehicle.get('NOx'),
            vehicle.get('PMx'),
            vehicle.get('fuel'),
            vehicle.get('electricity'),
            vehicle.get('noise'),
            vehicle.get('route'),
            vehicle.get('type'),
            vehicle.get('waiting'),
            vehicle.get('lane'),
            vehicle.get('pos'),
            vehicle.get('speed'),
            vehicle.get('angle'),
            vehicle.get('x'),
            vehicle.get('y')
        ]
        csv_writer.writerow(row)

# Close CSV file
csv_file.close()
