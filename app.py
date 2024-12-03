from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
from indeterminatebeam import Beam, Support, PointLoadV, TrapezoidalLoad
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/', methods=['GET', 'POST'])
def handle_data():
    # Extract input data from React
    length = request.json.get('length')
    point_loads = request.json.get('point_loads', [])
    distributed_loads = request.json.get('distributed_loads', [])
    supports = request.json.get('supports', [])
    youngmodules = request.json.get('youngmodules')
    area = request.json.get('area')
    inertia = request.json.get('inertia')
    
    # Create Beam instance
    beam = Beam(length, A=area, I=inertia, E=youngmodules)
    
    # Add supports to the beam
    for support in supports:
        position = support['position']
        support_type = support['type']
        if support_type == 'pinned':
            constraints = (1, 1, 0)
        elif support_type == 'roller':
            constraints = (0, 1, 0)
        elif support_type == 'fixed':
            constraints = (1, 1, 1)
        beam.add_supports(Support(position, constraints))
    
    # Add point loads to the beam
    for pl in point_loads:
        beam.add_loads(PointLoadV(pl['magnitude'] * -1, pl['position']))

    # Add distributed loads to the beam
    for dl in distributed_loads:
        # beam.add_loads(UDL(((dl['start_magnitude'] * -1, dl['end_magnitude'] * -1),(dl['start_position'], dl['end_position']))))
      
        # Add distributed load
        beam.add_loads(TrapezoidalLoad(force=(dl['start_magnitude']*-1, dl['end_magnitude']*-1), span=(dl['start_position'], dl['end_position'])))
    
    # Analyze the beam
    beam.analyse()
    
    # Extract reaction forces at support positions
    reactions = []
    for support in supports: 
        position = support['position']
        reaction_force = beam.get_reaction(x_coord=position)[1]
        reaction_momentum = beam.get_reaction(x_coord=position)[2]
        reactions.append({ 
            'position': position,
            'force': reaction_force,  # Consistent key naming
            'momentum': reaction_momentum  # Consistent key naming
        })
    
    # Create BeamDiagram array to include point loads, distributed loads, and supports
    BeamDiagram = []
 
    # Extract point loads data
    for pl in point_loads:
        pl_magnitude = pl['magnitude']
        pl_position = pl['position']
        
        BeamDiagram.append({
            'type': 'PointLoad',
            'magnitude': pl_magnitude,
            'position': pl_position
        })
    
    # Extract distributed loads data
    for dl in distributed_loads:
        dl_start_magnitude = dl['start_magnitude']
        dl_end_magnitude = dl['end_magnitude']
        dl_start_position = dl['start_position']
        dl_end_position = dl['end_position']
        
        BeamDiagram.append({
            'type': 'DistributedLoad',
            'start_magnitude': dl_start_magnitude,
            'end_magnitude': dl_end_magnitude,
            'start_position': dl_start_position,
            'end_position': dl_end_position
        })
    
    # Extract supports data
    for support in supports:
        position = support['position']
        support_type = support['type']
        if support_type == 'pinned':
            constraints = (1, 1, 0)
        elif support_type == 'roller':
            constraints = (0, 1, 0)
        elif support_type == 'fixed':
            constraints = (1, 1, 1)
        
        BeamDiagram.append({
            'type': 'Support',
            'position': position,
            'constraints': constraints,
            'support_type': support_type
        })
        
    positions = np.linspace(0, length, 100)
    deflections = [beam.get_deflection(x) for x in positions]
    shear_forces = [beam.get_shear_force(x) for x in positions]
    bending_moments = [beam.get_bending_moment(x) for x in positions]
    # Return supports, reaction forces, and BeamDiagram as a response
    response = {
        "Supports": supports,
        "reactions": reactions,
        "BeamDiagram": BeamDiagram,
        'deflection_data': [{'position': round(float(pos), 2), 'deflection': round(float(deflection), 6)} for pos, deflection in zip(positions, deflections)],
        'shear_force_data': [{'position': round(float(pos), 2), 'shear_force': round(float(shear_force), 2)} for pos, shear_force in zip(positions, shear_forces)],
        'bending_moment_data': [{'position': round(float(pos), 2), 'bending_moment': round(float(bending_moment), 2)} for pos, bending_moment in zip(positions, bending_moments)],
   }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
