import React from 'react'
import Img from '../../../images/Padeye-PadeyeFileOutputparameters-tensile-stress-area.png'
export default function Outputs(props) {

  return (
    <>
      <div>
        <div className={props.DesignLoads} style={{
          height: '45vw',
          left: '0'
        }}>
          <h3 className='text-aligh-center color-white'>ALLOWABLE STRESSES & DESIGN LOADS</h3>
          <br />
          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Allowable Stresses:</h3>
          <br />
          <div className='' style={{
            borderRadius: '10px',
          }}>      <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Allowable Bearing Stress</h3>
              <h3 className='sigma-symbol outer-sigma-symbol'>
                <span className='power' style={{ fontSize: '2vw', top: '-3px', color: 'white' }}>σ</span>
                <span className='LowerPower' style={{ color: 'white' }}>be(allow)</span>
              </h3>
              <div className='Calculator-Side-A outer-sigma-symbol-input'>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.AllowableBearingStressValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.AllowableBearingStressselectedUnit}
                    onChange={props.AllowableBearingStressselectedUnitOnchange}
                  >
                    {props.AllowableBearingStressOptions}
                  </select>
                </div>
              </div>
            </div>


            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Allowable Bending Stress
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (In-Plane)</span>
              </h3>
              <h3 className='sigma-symbol outer-sigma-symbol'>
                <span className='power' style={{ fontSize: '2vw', top: '-3px', color: 'white' }}>σ</span>
                <span className='LowerPower' style={{ color: 'white' }}>bd(allow)</span>
              </h3>
              <div className='Calculator-Side-A outer-sigma-symbol-input'>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.AllowableBendingStressInPlaneValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.AllowableBendingStressInPlaneselectedUnit}
                    onChange={props.AllowableBendingStressInPlaneselectedUnitOnchange}
                  >
                    {props.AllowableBendingStressInPlaneOption}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Allowable Bending Stress
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Out of-plane)</span>

              </h3>
              <h3 className='sigma-symbol outer-sigma-symbol'>
                <span className='power' style={{ fontSize: '2vw', top: '-3px', color: 'white' }}>σ</span>
                <span className='LowerPower' style={{ color: 'white' }}>bdo(allow)</span>
              </h3>
              <div className='Calculator-Side-A outer-sigma-symbol-input'>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.AllowableBendingStressOutofPlaneValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.AllowableBendingStressOutofPlaneselectedUnit}
                    onChange={props.AllowableBendingStressOutofPlaneselectedUnitOnchange}
                  >
                    {props.AllowableBendingStressOutofPlaneOption}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Allowable Tensile Stress</h3>
              <h3 className='sigma-symbol outer-sigma-symbol'>
                <span className='power' style={{ fontSize: '2vw', top: '-3px', color: 'white' }}>σ</span>
                <span className='LowerPower' style={{ color: 'white' }}>t(allow)</span>
              </h3>
              <div className='Calculator-Side-A outer-sigma-symbol-input'>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.AllowableTensileStressValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.AllowableTensileStressSelectedUnit}
                    onChange={props.AllowableTensileStressSelectedUnitOnchange}
                  >
                    {props.AllowableTensileStressOption}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Allowable Tensile Stress at pin hole</h3>
              <h3 className='sigma-symbol outer-sigma-symbol'>
                <span className='power' style={{ fontSize: '2vw', top: '-3px', color: 'white' }}>σ</span>
                <span className='LowerPower' style={{ color: 'white' }}>tp(allow)</span>
              </h3>
              <div className='Calculator-Side-A outer-sigma-symbol-input'>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.AllowableTensileStressatpinholeValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.AllowableTensileStressatpinholeSelectedUnit}
                    onChange={props.AllowableTensileStressatpinholeSelectedUnitOnchange}
                  >
                    {props.AllowableTensileStressatpinholeUnits}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Allowable Shear Stress</h3>
              <h3 className='sigma-symbol outer-sigma-symbol'>
                <span className='power' style={{ fontSize: '2vw', top: '-3px', color: 'white' }}>τ</span>
                <span className='LowerPower' style={{ color: 'white' }}>s(allow)</span>
              </h3>
              <div className='Calculator-Side-A outer-sigma-symbol-input'>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.AllowableShearStressValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.AllowableShearStressSelectedUnit}
                    onChange={props.AllowableShearStressSelectedUnitOnchange}
                  >
                    {props.AllowableShearStressOptions}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Allowable Hertz Stress
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Pin Hole)</span>
              </h3>
              <h3 className='sigma-symbol outer-sigma-symbol'>
                <span className='power' style={{ fontSize: '2vw', top: '-3px', color: 'white' }}>σ</span>
                <span className='LowerPower' style={{ color: 'white' }}>H(allow)</span>
              </h3>
              <div className='Calculator-Side-A outer-sigma-symbol-input'>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.AllowableHertzStressatPinHoleValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.AllowableHertzStressatPinHoleSelectedUnit}
                    onChange={props.AllowableHertzStressatPinHoleSelectedUnitOnchange}
                  >
                    {props.AllowableHertzStressatPinHoleOptions}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>

              <h3 className='claculator-conversation-title'>Allowable Weld Stress</h3>
              <h3 className='sigma-symbol outer-sigma-symbol'>
                <span className='power' style={{ fontSize: '2vw', top: '-3px', color: 'white' }}>σ</span>

                <span className='LowerPower' style={{ color: 'white' }}>w(allow)</span></h3>
              <div className='Calculator-Side-A outer-sigma-symbol-input'>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.AllowableWeldStressValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.AllowableWeldStressSelectedUnit}
                    onChange={props.AllowableWeldStressSelectedUnitOnchange}
                  >
                    {props.AllowableWeldStressOptions}
                  </select>
                </div>
              </div>
            </div>
            <br />
            <br />
            <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Design Loads:</h3>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>

              <h3 className='claculator-conversation-title'>Load in Vertical Direction
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Z direction)</span>
              </h3>
              <div className='Calculator-Side-A '>
                <h3 className='sigma-symbol'>
                  <span className='power' style={{ fontSize: '2vw', top: '-3px' }}>P</span>

                  <span className='LowerPower'>z</span></h3>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.LoadinVerticalDirectionZdirectionValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.LoadinVerticalDirectionZdirectionSelectedUnit}
                    onChange={props.handleLoadinVerticalDirectionZdirectionselectedUnitChange}
                  >
                    {props.LoadinVerticalDirectionZdirectionOptions}
                  </select>
                </div>
              </div>
            </div>



            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Load in Horizontal Direction
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Y direction)</span>
              </h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>
                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>P</span>
                  <span className='LowerPower'>y</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.LoadinHorizontalDirectionYdirectionValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.LoadinHorizontalDirectionYdirectionSelectedUnit}
                    onChange={props.LoadinHorizontalDirectionYdirectionSelectedUnitOnchange}
                  >
                    {props.LoadinHorizontalDirectionYdirectionOptions}
                  </select>
                </div>
              </div>
            </div>



            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Out-of-plane Load
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (X direction)</span>
              </h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>
                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>P</span>
                  <span className='LowerPower'>x</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.OutofplaneLoadLateralLoadXdirectionValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.OutofplaneLoadLateralLoadXdirectionSelectedUnit}
                    onChange={props.OutofplaneLoadLateralLoadXdirectionSelectedUnitOnchange}
                  >
                    {props.OutofplaneLoadLateralLoadXdirectionOptions}
                  </select>
                </div>
              </div>
            </div>


            {/* PadEyeDesignLoad   */}
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>
                Pad-eye Design Load
              </h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>
                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>P</span>
                  <span className='LowerPower'>d</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.PadEyeDesignLoadvalue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.PadEyeDesignLoadSelectedUnit}
                    onChange={props.PadEyeDesignLoadSelectedUnitOnchange}
                  >
                    {props.PadEyeDesignLoadOptions}
                  </select>
                </div>
              </div>
            </div>
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>
                Design Load in Vertical Direction
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Z-direction)</span>
              </h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>
                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>P</span>
                  <span className='LowerPower'>zd</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.DesignLoadinVerticalDirectionZdirectionvalue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.DesignLoadinVerticalDirectionZdirectionSelectedUnit}
                    onChange={props.DesignLoadinVerticalDirectionZdirectionSelectedUnitOnchange}
                  >
                    {props.DesignLoadinVerticalDirectionZdirectionOptions}
                  </select>
                </div>
              </div>
            </div>
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>
                Design Load in Horizontal Direction
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Y-direction)</span>
              </h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>
                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>P</span>
                  <span className='LowerPower'>yd</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.DesignLoadinHorizontalDirectionYdirectionValue}
                    onChange={props.LoadinVerticalDirectionZdirectionValueOnchange}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.DesignLoadinHorizontalDirectionYdirectionSelectedUnit}
                    onChange={props.DesignLoadinHorizontalDirectionYdirectionSelectedUnitOnchange}
                  >
                    {props.DesignLoadinHorizontalDirectionYdirectionOption}
                  </select>
                </div>
              </div>
            </div>
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>
                Design Load Out-of-Plane
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Lateral, X-direction)</span>

              </h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>
                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>P</span>
                  <span className='LowerPower'>xd</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.DesignLoadOutofPlaneLateralXdirectionValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.DesignLoadOutofPlaneLateralXdirectionSelectedUnit}
                    onChange={props.DesignLoadOutofPlaneLateralXdirectionSelectedUnitOnchange}
                  >
                    {props.DesignLoadOutofPlaneLateralXdirectionOption}
                  </select>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className={props.GEOMETRYCHECK} style={{
          height: '20vw',
          left: '50%',
          top: '115%',
          width: '42vw',
        }}>
          <h3 className='text-aligh-center color-white'>GEOMETRY CHECK</h3>
          <br />
          <div className='' style={{
            borderRadius: '10px',
          }}>
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Clearance between pin hole dia and pin dia</h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>
                  <span className='power' style={{ fontSize: '1.5vw', top: '-1px' }}>δD</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.Clearancebetweenpinholediaandpindiavalue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.ClearancebetweenpinholediaandpindiaSelectedunit}
                    onChange={props.ClearancebetweenpinholediaandpindiaSelectedunitOnchange}
                  >
                    {props.ClearancebetweenpinholediaandpindiaOptions}
                  </select>
                </div>
              </div>
            </div>
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Length Clearance of Shackle</h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol sigmawithshorterlowerPower'>
                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>δL</span>
                  <span className='LowerPower'>shackle</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.LengthClearanceofShackleValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.LengthClearanceofShackleselectedunit}
                    onChange={props.LengthClearanceofShackleselectedunitOnchange}
                  >
                    {props.LengthClearanceofShackleOption}
                  </select>
                </div>
              </div>
            </div>
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Width Clearance of Shackle</h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol sigmawithshorterlowerPower'>

                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>δW</span>
                  <span className='LowerPower'>shackle</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.WidthClearanceofShackle}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.WidthClearanceofShackleSelectedUnit}
                    onChange={props.WidthClearanceofShackleSelectedUnitOnchange}
                  >
                    {props.WidthClearanceofShackleOption}
                  </select>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className={props.WELDCHECKATCHEEKPLATEWELD} style={{
          height: '20vw',
          left: '50%',
          width: '42vw',
          top: '201%',
        }}>
          <h3 className='text-aligh-center color-white'>WELD CHECK AT CHEEK PLATE WELD</h3>
          <br />
          <br />

          <div className=''>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Throat Thickness</h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>

                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>t</span>
                  <span className='LowerPower'>wtc</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.WeldCheckatCheekPlateWeldThroatthicknessValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.WeldCheckatCheekPlateWeldThroatthicknessSelectedUnit}
                    onChange={props.WeldCheckatCheekPlateWeldThroatthicknessSelectedUnitOnchange}
                  >
                    {props.WeldCheckatCheekPlateWeldThroatthicknessOption}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Weld Length</h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>

                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>L</span>
                  <span className='LowerPower'>wc</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.WeldLength}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.WeldLengthSelectedUnit}
                    onChange={props.WeldLengthSelectedUnitOnchange}
                  >
                    {props.WeldLengthOptions}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Weld Area at Cheek Plate</h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>

                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>A</span>
                  <span className='LowerPower'>wc</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.WeldAreaatCheekPlate}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.WeldAreaatCheekPlateSelectedUnit}
                    onChange={props.WeldAreaatCheekPlateSelectedUnitOnchange}
                  >
                    {props.WeldAreaatCheekPlateOptions}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Design Load on one cheek plate</h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>

                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>P</span>
                  <span className='LowerPower'>dc</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.DesignLoadononecheekplate}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.DesignLoadononecheekplateSelectedUnit}
                    onChange={props.DesignLoadononecheekplateSelectedUnitOnchange}
                  >
                    {props.DesignLoadononecheekplateOptions}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Weld Stress at Cheek Weld
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act)</span>
              </h3>
              <div className='Calculator-Side-A '>
                <h3 className='sigma-symbol '>
                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>σ</span>
                  <span className='LowerPower'>wc</span>
                </h3>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.WeldStressatCheekWeldActual}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.WeldStressatCheekWeldActualSelectedUnit}
                    onChange={props.WeldStressatCheekWeldActualSelectedUnitOnchange}
                  >
                    {props.WeldStressatCheekWeldActualoptions}
                  </select>
                </div>
              </div>
            </div>
          </div>
        </div>


        <div className={props.STRESSCHECKSATPINHOLE} style={{
          height: '45vw',
          left: '0%',
          width: '38vw',
          top: '290%',
          zIndex: '1'
        }}>
          <h3 className='text-aligh-center color-white'>STRESS CHECKS AT PIN HOLE</h3>
          <br />
          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Bearing Stress Check at Pin Hole:</h3>
          <br />
          <div className=''>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Bearing Area</h3>
              <div className='Calculator-Side-A'>
                <h3 className='sigma-symbol'>

                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>A</span>
                  <span className='LowerPower'>be</span>
                </h3>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.BearingAreaValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.BearingAreaSelectedUnit}
                    onChange={props.BearingAreaSelectedUnitOnchange}
                  >
                    {props.BearingAreaOptions}
                  </select>
                </div>
              </div>
            </div>
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Bearing Stress
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
              </h3>
              <div className='Calculator-Side-A ' style={{ width: '40%' }}>
                <h3 className='sigma-symbol '>
                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>σ</span>
                  <span className='LowerPower'>be</span>
                </h3>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.BearingStressActualValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.BearingStressActualSelectedUnit}
                    onChange={props.BearingStressActualSelectedUnitOnchange}
                  >
                    {props.BearingStressActualOptions}
                  </select>
                </div>
              </div>
            </div>

            <br />
            <br />
            <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Shear Stress Check at Pin Hole:</h3>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Radius of Cheek Plate</h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>

                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>R</span>
                  <span className='LowerPower'>c</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.RadiusofCheekPlateValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.RadiusofCheekPlateSelectedUnit}
                    onChange={props.RadiusofCheekPlateSelectedUnitOnchange}
                  >
                    {props.RadiusofCheekPlateOptions}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Shear Area of pin hole</h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>

                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>A</span>
                  <span className='LowerPower'>sp</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.ShearAreaofpinholeValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.ShearAreaofpinholeSelectedUnit}
                    onChange={props.ShearAreaofpinholeSelectedUnitOnchange}
                  >
                    {props.ShearAreaofpinholeOption}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Shear Stress at pin hole</h3>
              <h3 className='sigma-symbol outer-sigma-symbol'>

                <span className='power' style={{ fontSize: '1.5vw', top: '-3px', color: 'white' }}>σ</span>
                <span className='LowerPower' style={{ color: 'white' }}>sp(act)</span>
              </h3>
              <div className='Calculator-Side-A outer-sigma-symbol-input'>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.ShearStressatpinholeValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.ShearStressatpinholeSelectedUnit}
                    onChange={props.ShearStressatpinholeSelectedUnitOnchange}
                  >
                    {props.ShearStressatpinholeOptions}
                  </select>
                </div>
              </div>
            </div>

            <br />
            <br />
            <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Tensile Stress Check at Pin Hole:</h3>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title' >Tensile Area for section
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> A-A </span>
              </h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>

                  <span className='power' style={{ fontSize: '1vw !important', top: '-3px' }}>A</span>
                  <span className='LowerPower'>t1p</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.TensileAreaforsectionAAValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.TensileAreaforsectionAASelectedUnit}
                    onChange={props.TensileAreaforsectionAASelectedUnitOnchange}
                  >
                    {props.TensileAreaforsectionAAOption}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title' style={{ fontSize: '1.1vw' }}>Tensile Stress at pin hole
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
                at section
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> A-A </span>
              </h3>
              <div className='Calculator-Side-A '>
                <h3 className='sigma-symbol '>
                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>σ</span>
                  <span className='LowerPower' style={{ position: 'relative', top: '0px', }}>tp</span>
                </h3>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.TensileStressatpinholeActualatsectionAAValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.TensileStressatpinholeActualatsectionAASelectedUnit}
                    onChange={props.TensileStressatpinholeActualatsectionAASelectedUnitOnchange}
                  >
                    {props.TensileStressatpinholeActualatsectionAAOption}
                  </select>
                </div>
              </div>
            </div>



            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Tensile Area for section
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> B-B </span>
              </h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>

                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>A</span>
                  <span className='LowerPower'>t2p</span>
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.TensileAreaforsectionBBValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.TensileAreaforsectionBBSelectedUnit}
                    onChange={props.TensileAreaforsectionBBSelectedUnitOnchange}
                  >
                    {props.TensileAreaforsectionBBOption}
                  </select>
                </div>
              </div>
            </div>



            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title' style={{ fontSize: '1.1vw' }}>Tensile Stress at pin hole
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
                at section
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> B-B </span>
              </h3>
              <div className='Calculator-Side-A'>
                <h3 className='sigma-symbol '>
                  <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>σ</span>
                  <span className='LowerPower' style={{ position: 'relative', top: '0px', }}>tp</span>
                </h3>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.TensileStressatpinholeActualatsectionBBValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.TensileStressatpinholeActualatsectionBBSelectedUnit}
                    onChange={props.TensileStressatpinholeActualatsectionBBSelectedUnitOnchange}
                  >
                    {props.TensileStressatpinholeActualatsectionBBOption}
                  </select>
                </div>
              </div>
            </div>

            <br />
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <img style={{ width: '20vw', height: '18vw' }} src={Img} alt='' />
            </div>
            <br />
            <br />
            <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Hertz/Contact Stress Check at Pin Hole:</h3>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Design Load per unit Length</h3>
              <div className='Calculator-Side-A'>
                <br />
                <h3 className='sigma-symbol'>
                  P
                </h3>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.DesignLoadperunitLengthValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.DesignLoadperunitLengthSelectedUnit}
                    onChange={props.DesignLoadperunitLengthSelectedUnitOnchange}
                  >
                    {props.DesignLoadperunitLengthOptions}
                  </select>
                </div>
              </div>
            </div>


            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <h3 className='claculator-conversation-title'>Hertz Stress at Pin Hole</h3>
              <h3 className='sigma-symbol outer-sigma-symbol'>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px', color: 'white' }}>σ</span>
                <span className='LowerPower' style={{ color: 'white' }}>H(act)</span>
              </h3>
              <div className='Calculator-Side-A outer-sigma-symbol-input'>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={props.HertzStressatPinHoleValue}
                    readOnly
                  />
                  <select
                    className='Calculator-select-option'
                    value={props.HertzStressatPinHoleSelectedUnit}
                    onChange={props.HertzStressatPinHoleSelectedUnitOnchange}
                  >
                    {props.HertzStressatPinHoleOptions}
                  </select>
                </div>
              </div>
            </div>

          </div>
        </div>



        <div className={props.WELDSTRESSCHECKOFBASEWELD} style={{
          height: '45vw',
          left: '0%',
          width: '38vw',
          top: '465%',
          zIndex: ' 1',
        }}>
          <h3 className='text-aligh-center color-white'>WELD STRESS CHECK OF BASE WELD</h3>
          <br />
          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Tensile Stress Check:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Throat Thickness</h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol'>

                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>t</span>
                <span className='LowerPower'>wt</span>
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.ThroatthicknessValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.ThroatthicknessSelectedUnit}
                  onChange={props.ThroatthicknessSelectedUnitOnchange}
                >
                  {props.ThroatthicknessOption}
                </select>
              </div>
            </div>
          </div>


          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Total Weld Length</h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol'>

                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>L</span>
                <span className='LowerPower'>w</span>
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.TotalWeldLengthvalue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.TotalWeldLengthSelectedUnit}
                  onChange={props.TotalWeldLengthSelectedUnitOnchange}
                >
                  {props.TotalWeldLengthOption}
                </select>
              </div>
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Moment lever for Horizontal/Lateral Force</h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol'>
                h
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.MomentleverforHorizontalLateralForceValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.MomentleverforHorizontalLateralForceSelectedUnit}
                  onChange={props.MomentleverforHorizontalLateralForceSelectedUnitOnchange}
                >
                  {props.MomentleverforHorizontalLateralForceOptions}
                </select>
              </div>
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Moment lever for Vertical Force</h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol'>
                e
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.MomentleverforVerticalForceValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.MomentleverforVerticalForceSelectedUnit}
                  onChange={props.MomentleverforVerticalForceSelectedUnitOnchange}
                >
                  {props.MomentleverforVerticalForceOptions}
                </select>
              </div>
            </div>
          </div>

          <br />
          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Shear Stress at Base Weld:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Shear Area of Base Weld</h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol'>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>A</span>
                <span className='LowerPower'>sbw</span>
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.ShearStressatBaseWeldValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.ShearStressatBaseWeldSelectedUnit}
                  onChange={props.ShearStressatBaseWeldSelectedUnitOnchange}
                >
                  {props.ShearStressatBaseWeldOption}
                </select>
              </div>
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Shear Stress
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
              - Horizontal
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Y -direction) </span>
            </h3>
            <div className='Calculator-Side-A '>
              <h3 className='sigma-symbol '>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>τ</span>
                <span className='LowerPower'>sbwy</span>
              </h3>
              <br />
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.ShearStressatBaseWeldActualHorizontalYdirectionValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.ShearStressatBaseWeldActualHorizontalYdirectionSelectedUnit}
                  onChange={props.ShearStressatBaseWeldActualHorizontalYdirectionSelectedUnitOnchange}
                >
                  {props.ShearStressatBaseWeldActualHorizontalYdirectionOptions}
                </select>
              </div>
            </div>
          </div>


          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Shear Stress
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
              - Lateral
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (X - direction) </span>
            </h3>
            <div className='Calculator-Side-A '>
              <h3 className='sigma-symbol '>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>τ</span>
                <span className='LowerPower'>sbwx</span>
              </h3>
              <br />
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.ShearStressatBaseWeldActualLateralXdirectionValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.ShearStressatBaseWeldActualLateralXdirectionSelectedUnit}
                  onChange={props.ShearStressatBaseWeldActualLateralXdirectionSelectedUnitOnchange}
                >
                  {props.ShearStressatBaseWeldActualLateralXdirectionOption}
                </select>
              </div>
            </div>
          </div>
          <br />
          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Tensile Stress at Base Weld:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Tensile Area at Base
            </h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol'>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>A</span>
                <span className='LowerPower'>tbw</span>
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.ShearStressatBaseWeldValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.ShearStressatBaseWeldSelectedUnit}
                  onChange={props.ShearStressatBaseWeldSelectedUnitOnchange}
                >
                  {props.ShearStressatBaseWeldOption}
                </select>
              </div>
            </div>
          </div>


          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Tensile Stress at Base Weld
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act)</span>
            </h3>
            <div className='Calculator-Side-A ' >
              <h3 className='sigma-symbol '>
                <span className='power' style={{ fontSize: '1vw !important', top: '-3px' }}>σ</span>
                <span className='LowerPower'>tbw</span>
              </h3>
              <br />
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.TensileStressatBaseWeldActual}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.TensileStressatBaseWeldActualSelectedUnit}
                  onChange={props.TensileStressatBaseWeldActualSelectedUnitOnchange}
                >
                  {props.TensileStressatBaseWeldActualoption}
                </select>
              </div>
            </div>
          </div>


          <br />
          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Section Modulus of Base Weld:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Section Modulus of Weld about X-axis
            </h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol sigmawithshorterlowerPower'>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>Z</span>
                <span className='LowerPower'>wbxx</span>
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.SectionModulusofWeldaboutXaxis}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.SectionModulusofWeldaboutXaxisSelectedUnit}
                  onChange={props.SectionModulusofWeldaboutXaxisSelectedUnitOnchange}
                >
                  {props.SectionModulusofWeldaboutXaxisOptions}
                </select>
              </div>
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Section Modulus of Weld about Y-axis
            </h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol sigmawithshorterlowerPower'>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>Z</span>
                <span className='LowerPower'>wbyy</span>
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.SectionModulusofWeldaboutYaxis}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.SectionModulusofWeldaboutYaxisSelectedUnit}
                  onChange={props.SectionModulusofWeldaboutYaxisSelectedUnitOnchange}
                >
                  {props.SectionModulusofWeldaboutYaxisOptions}
                </select>
              </div>
            </div>
          </div>

          <br />
          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Bending Stress Check at Base Weld:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Total Design Moment about X-axis
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (In-plane)</span>
            </h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol sigmawithshorterlowerPower'>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>M</span>
                <span className='LowerPower'>dxx</span>
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.TotalDesignMomentaboutXaxisInplaneValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.TotalDesignMomentaboutXaxisInplaneSelectedUnit}
                  onChange={props.TotalDesignMomentaboutXaxisInplaneSelectedUnitOnchange}
                >
                  {props.TotalDesignMomentaboutXaxisInplaneOptions}
                </select>
              </div>
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Bending Stress
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
              - about X Axis
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}>  (In Plane) </span>
            </h3>
            <div className='Calculator-Side-A '>
              <h3 className='sigma-symbol '>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>σ</span>
                <span className='LowerPower'>bbwx</span>
              </h3>
              <br />
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.BendingStressatBaseWeldActualaboutXAxisInPlane}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.BendingStressatBaseWeldActualaboutXAxisInPlaneSelectedUnit}
                  onChange={props.BendingStressatBaseWeldActualaboutXAxisInPlaneSelectedUnitOnchange}
                >
                  {props.BendingStressatBaseWeldActualaboutXAxisInPlaneOptions}
                </select>
              </div>
            </div>
          </div>


          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Total Design Moment about Y-axis
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Out-of Plane) </span>
            </h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol sigmawithshorterlowerPower'>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>M</span>
                <span className='LowerPower'>dyy</span>
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.TotalDesignMomentaboutYaxisOutofPlaneValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.TotalDesignMomentaboutYaxisOutofPlaneSelectedUnits}
                  onChange={props.TotalDesignMomentaboutYaxisOutofPlaneSelectedUnitsOnchange}
                >
                  {props.TotalDesignMomentaboutYaxisOutofPlaneOption}
                </select>
              </div>
            </div>
          </div>




          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Bending Stress
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
              - about Y-axis
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Out-of Plane) </span>
            </h3>
            <div className='Calculator-Side-A'>
              <h3 className='sigma-symbol '>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>σ</span>
                <span className='LowerPower'>bby</span>
              </h3>
              <br />
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.BendingStressaboutYaxisOutofPlane}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.BendingStressaboutYaxisOutofPlaneSelectedUnit}
                  onChange={props.BendingStressaboutYaxisOutofPlaneSelectedUnitOnchange}
                >
                  {props.BendingStressaboutYaxisOutofPlaneOptions}
                </select>
              </div>
            </div>
          </div>

          <br />
          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Total Stress Check at Base Weld:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Total Stress at Base Weld
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
            </h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol sigmawithshorterlowerPower'>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>M</span>
                <span className='LowerPower'>dyy</span>
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.TotalStressatBaseWeldActual}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.TotalStressatBaseWeldActualSelectedUnit}
                  onChange={props.TotalStressatBaseWeldActualSelectedUnitOnchange}
                >
                  {props.TotalStressatBaseWeldActualOptions}
                </select>
              </div>
            </div>
          </div>
        </div>

        <div className={props.STRESSCHECKSATBASEPLATE} style={{
          height: '45vw',
          left: '52%',
          width: '38vw',
          top: '290%',
          zIndex: ' 0',
        }}>
          <h3 className='text-aligh-center color-white'>STRESS CHECKS AT BASE PLATE</h3>
          <br />
          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Tensile Stress Check:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Tensile Area</h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol'>

                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>A</span>
                <span className='LowerPower'>tb</span>
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.TensileAreaValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.TensileAreaSelectedUnit}
                  onChange={props.TensileAreaSelectedUnitOnchange}
                >
                  {props.TensileAreaOptions}
                </select>
              </div>
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Tensile Stress at Base
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}>(Act)</span>
            </h3>
            <div className='Calculator-Side-A '>
              <h3 className='sigma-symbol '>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>σ</span>
                <span className='LowerPower'>tb</span>
              </h3>
              <br />
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.TensileStressatBaseActualValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.TensileStressatBaseActualSelectedUnit}
                  onChange={props.TensileStressatBaseActualSelectedUnitOnchange}
                >
                  {props.TensileStressatBaseActualOption}
                </select>
              </div>
            </div>
          </div>
          <br />
          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Bending Stress Check:</h3>
          <br />
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Moment lever for Horizontal/Lateral Force</h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol'>
                h
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.MomentleverforHorizontalLateralForceValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.MomentleverforHorizontalLateralForceSelectedUnit}
                  onChange={props.MomentleverforHorizontalLateralForceSelectedUnitOnchange}
                >
                  {props.MomentleverforHorizontalLateralForceOptions}
                </select>
              </div>
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Moment lever for Vertical Force</h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol'>
                e
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.MomentleverforVerticalForceValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.MomentleverforVerticalForceSelectedUnit}
                  onChange={props.MomentleverforVerticalForceSelectedUnitOnchange}
                >
                  {props.MomentleverforVerticalForceOptions}
                </select>
              </div>
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Total Design Moment about X-axis
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (In-plane)</span>
            </h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol sigmawithshorterlowerPower'>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>M</span>
                <span className='LowerPower'>dxx</span>
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.TotalDesignMomentaboutXaxisInplaneValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.TotalDesignMomentaboutXaxisInplaneSelectedUnit}
                  onChange={props.TotalDesignMomentaboutXaxisInplaneSelectedUnitOnchange}
                >
                  {props.TotalDesignMomentaboutXaxisInplaneOptions}
                </select>
              </div>
            </div>
          </div>



          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Section Modulus of Base about X-axis</h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol sigmawithshorterlowerPower'>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>Z</span>
                <span className='LowerPower'>dxx</span>
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.SectionModulusofBaseaboutXaxisValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.SectionModulusofBaseaboutXaxisSelectedUnit}
                  onChange={props.SectionModulusofBaseaboutXaxisSelectedUnitOnchange}
                >
                  {props.SectionModulusofBaseaboutXaxisOption}
                </select>
              </div>
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Bending Stress
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act)</span>
              - about X-axis
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (In-plane)</span>
            </h3>
            <div className='Calculator-Side-A '>
              <h3 className='sigma-symbol '>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>σ</span>
                <span className='LowerPower'>bbx</span>
              </h3>
              <br />
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.BendingStressActualaboutXaxisInplaneValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.BendingStressActualaboutXaxisInplaneSelectedUnit}
                  onChange={props.BendingStressActualaboutXaxisInplaneOnchange}
                >
                  {props.BendingStressActualaboutXaxisInplaneOption}
                </select>
              </div>
            </div>
          </div>


          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Total Design Moment about Y-axis
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Out-of Plane)</span>
            </h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol sigmawithshorterlowerPower'>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>M</span>
                <span className='LowerPower'>dyy</span>
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.TotalDesignMomentaboutYaxisOutofPlaneValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.TotalDesignMomentaboutYaxisOutofPlaneSelectedUnit}
                  onChange={props.TotalDesignMomentaboutYaxisOutofPlaneSelectedUnitOnchange}
                >
                  {props.TotalDesignMomentaboutYaxisOutofPlaneOption}
                </select>
              </div>
            </div>
          </div>



          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Section Modulus of Base about Y-axis</h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol'>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>Z</span>
                <span className='LowerPower'>byy</span>
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.SectionModulusofBaseaboutYaxisValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.SectionModulusofBaseaboutYaxisSelectedUnit}
                  onChange={props.SectionModulusofBaseaboutYaxisSelectedUnitOnchange}
                >
                  {props.SectionModulusofBaseaboutYaxisOption}
                </select>
              </div>
            </div>
          </div>




          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Bending Stress
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act)</span>
              - about Y-axis
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Out-of-Plane)</span>
            </h3>
            <div className='Calculator-Side-A '>
              <h3 className='sigma-symbol '>
                <span className='power' style={{ fontSize: '1vw !important', top: '-3px' }}>σ</span>
                <span className='LowerPower'>bby</span>
              </h3>
              <br />
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.BendingStressActualaboutYaxisOutofPlaneValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.BendingStressActualaboutYaxisOutofPlaneSelectedUnit}
                  onChange={props.BendingStressActualaboutYaxisOutofPlaneSelectedUnitOnchange}
                >
                  {props.BendingStressActualaboutYaxisOutofPlaneOption}
                </select>
              </div>
            </div>
          </div>

          <br />
          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Shear Stress Check at Base:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Shear Area at Base
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Horizontal)</span>
            </h3>
            <div className='Calculator-Side-A'>
              <br />
              <h3 className='sigma-symbol'>

                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>A</span>
                <span className='LowerPower'>sb</span>
              </h3>
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.ShearAreaatBaseHorizontalValue}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.ShearAreaatBaseHorizontalSelectedUnit}
                  onChange={props.ShearAreaatBaseHorizontalSelectedUnitOnchange}
                >
                  {props.ShearAreaatBaseHorizontalOption}
                </select>
              </div>
            </div>
          </div>


          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Shear Stress at Base
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
              - Horizontal
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Y direction)</span>
            </h3>
            <div className='Calculator-Side-A '>
              <h3 className='sigma-symbol '>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>τ</span>
                <span className='LowerPower'>sby</span>
              </h3>
              <br />
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.ShearStressatBaseActualHorizontalYdirection}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.ShearStressatBaseActualHorizontalYdirectionSelectedUnit}
                  onChange={props.ShearStressatBaseActualHorizontalYdirectionSelectedUnitOnchange}
                >
                  {props.ShearStressatBaseActualHorizontalYdirectionOptions}
                </select>
              </div>
            </div>
          </div>


          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Shear Stress at Base
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}>(Act)</span>
              - Lateral
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}>(X direction)</span>
            </h3>
            <div className='Calculator-Side-A'>
              <h3 className='sigma-symbol'>

                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>τ</span>
                <span className='LowerPower'>sbx(act)</span>
              </h3>
              <br />
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.ShearStressatBaseActualLateralXdirection}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.ShearStressatBaseActualLateralXdirectionSelectedUnit}
                  onChange={props.ShearStressatBaseActualLateralXdirectionSelectedUnitOnchange}
                >
                  {props.ShearStressatBaseActualLateralXdirectionOption}
                </select>
              </div>
            </div>
          </div>

          <br />
          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Von Mises Stress Check at Base:</h3>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Von Mises Stress at Base
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act)</span>
            </h3>
            <div className='Calculator-Side-A '>
              <h3 className='sigma-symbol '>
                <span className='power' style={{ fontSize: '1.5vw', top: '-3px' }}>τ</span>
                <span className='LowerPower'>sbx</span>
              </h3>
              <br />
              <div className='input-and-select-div'>
                <input
                  className='calculator-input'
                  type="number"
                  value={props.VonMisesStressatBaseActual}
                  readOnly
                />
                <select
                  className='Calculator-select-option'
                  value={props.VonMisesStressatBaseActualSelectedUnit}
                  onChange={props.VonMisesStressatBaseActualSelectedUnitOnchange}
                >
                  {props.VonMisesStressatBaseActualOption}
                </select>
              </div>
            </div>
          </div>
        </div>
        <div className={props.FinalChecks} style={{
          height: '45vw',
          left: '52%',
          width: '38vw',
          top: '465%',
        }}>
          <h3 className='text-aligh-center color-white'>FINAL CHECKS</h3>
          <br />
          <br />
          <h3 className='color-white Finalchecks' style={{ fontSize: '1.8vw' }}>Geometry check:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title Finalcheck'>
              Main Plate Radius
            </h3>
            <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
              {props.MainPlateRadius}
            </div>
          </div>
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title Finalcheck'>
              Shackle Clearance Checks
            </h3>
            <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
              {props.ShackleClearanceChecks}
            </div>
          </div>
          <br />
          <h3 className='color-white Finalchecks' style={{ fontSize: '1.8vw' }}>Stress Checks at Pin Hole:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title Finalcheck'>
              Bearing Stress Check at Pin Hole
            </h3>
            <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
              {props.BearingStressCheckatPinHole}
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title Finalcheck'>
              Shear Stress Check at Pin Hole
            </h3>
            <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
              {props.ShearAreaofPinHole}
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title Finalcheck'>
              Tensile Stress Check at Pin Hole
            </h3>
            <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
              {props.TensileStressCheckatPinHole}
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title Finalcheck'>
              Hertz/Contact Stress Check at Pin Hole
            </h3>
            <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
              {props.HertzContactStressCheckatPinHole}
            </div>
          </div>
          <br />
          <h3 className='color-white Finalchecks' style={{ fontSize: '1.8vw' }}>Stress Check Base Plate:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title Finalcheck'>
              Tensile Stress Check at Base
            </h3>
            <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
              {props.TensileStressCheckatBase}
            </div>
          </div>


          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title Finalcheck'>
              Bending Stress Check at Base
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (X-axis) </span>
            </h3>
            <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
              {props.BendingStressCheckatBaseXaxis}
            </div>
          </div>
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title Finalcheck'>
              Bending Stress Check at Base
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Y-axis) </span>
            </h3>
            <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
              {props.BendingStressCheckatBaseYaxis}
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title Finalcheck'>
              Shear Stress Check at Base
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Y direction) </span>
            </h3>
            <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
              {props.ShearStressCheckatBaseYdirection}
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title Finalcheck'>
              Shear Stress Check at Base
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (X direction) </span>
            </h3>
            <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
              {props.ShearStressCheckatBaseXdirection}
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title Finalcheck'>
              Combined Stress Check as per AISC H2-1
            </h3>
            <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
              {props.combinedStressCheckasperAISCH21Condition}
            </div>
          </div>

          <br />
          <h3 className='color-white Finalchecks' style={{ fontSize: '1.8vw' }}>Weld Stress Check of Base Weld:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title Finalcheck'>
              Combined Stress Check at Base
            </h3>
            <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
              {props.CombinedStressCheckatBaseCondition}
            </div>
          </div>

          <br />
          <h3 className='color-white Finalchecks' style={{ fontSize: '1.8vw' }}>Weld Check at Cheek Plate Weld:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title Finalcheck'>
              Stress Check at Cheek Plate Weld
            </h3>
            <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
              {props.StressCheckatCheekPlateWeld}
            </div>
          </div>

        </div>

        {/* <div className={DropDowmThirdrightMain} style={{
          height: '45vw',
          left: '52%',
          width: '38vw',
          top: '465%',
        }}>
          <h3 className='text-aligh-center color-white Finalchecks'>FINAL CHECKS</h3>
          <br />
          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Geometry check:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Main Plate Radius
            </h3>
            <div className='Calculator-Side-A' style={{ width: '15%', justifyContent: 'center' }}>
              {props.MainPlateRadiusCondition}
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Shackle Clearance Checks
            </h3>
            <div className='Calculator-Side-A' style={{ width: '15%', justifyContent: 'center' }}>
              {props.ShackleClearanceChecksCondition}
            </div>
          </div>


          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Stress Checks at Pin Hole:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Bearing Stress Check at Pin Hole
            </h3>
            <div className='Calculator-Side-A' style={{ width: '15%', justifyContent: 'center' }}>
              {props.BearingStressCheckatPinHoleCondition}
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Shear Stress Check at Pin Hole
            </h3>
            <div className='Calculator-Side-A' style={{ width: '15%', justifyContent: 'center' }}>
              {props.ShearStressCheckatPinHoleCondition}
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Tensile Stress Check at Pin Hole
            </h3>
            <div className='Calculator-Side-A' style={{ width: '15%', justifyContent: 'center' }}>
              {props.TensileStressCheckatPinHoleCondition}
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Hertz/Contact Stress Check at Pin Hole
            </h3>
            <div className='Calculator-Side-A' style={{ width: '15%', justifyContent: 'center' }}>
              {props.HertzContactStressCheckatPinHoleCondition}
            </div>
          </div>


          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Stress Check Base Plate:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Tensile Stress Check at Base
            </h3>
            <div className='Calculator-Side-A' style={{ width: '15%', justifyContent: 'center' }}>
              {props.TensileStressCheckatBaseCondition}
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Bending Stress Check at Base
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (X-axis) </span>
            </h3>
            {/* {props.BendingStressActualaboutXaxisInplaneValue} */}
        {/* {props.AllowableBendingStressInPlaneValue}  
            <div className='Calculator-Side-A' style={{ width: '15%', justifyContent: 'center' }}>
              {props.BendingStressCheckatBaseXaxisCondition}
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Bending Stress Check at Base
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Y-axis) </span>
            </h3>
            <div className='Calculator-Side-A' style={{ width: '15%', justifyContent: 'center' }}>
              {props.BendingStressCheckatBaseYaxisCondition}
            </div>
          </div>


          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Shear Stress Check at Base
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Y direction) </span>
            </h3>
            <div className='Calculator-Side-A' style={{ width: '15%', justifyContent: 'center' }}>
              {props.ShearStressCheckatBaseYdirectionCondition}
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Shear Stress Check at Base
              <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (X direction) </span>
            </h3>
            <div className='Calculator-Side-A' style={{ width: '15%', justifyContent: 'center' }}>
              {props.ShearStressCheckatBaseXdirectionCondition}
            </div>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Combined Stress Check as per AISC H2-1
            </h3>
            <div className='Calculator-Side-A' style={{ width: '15%', justifyContent: 'center' }}>
              {props.CombinedStressCheckatStressCheckBasePlateCondition}
            </div>
          </div>

          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Weld Stress Check of Base Weld:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Combined Stress Check at Base
            </h3>
            <div className='Calculator-Side-A' style={{ width: '15%', justifyContent: 'center' }}>
              {props.CombinedStressCheckatBaseCondition}
            </div>
          </div>

          <br />
          <h3 className='color-white' style={{ fontSize: '1.8vw' }}>Weld Check at Cheek Plate Weld:</h3>
          <br />

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>
              Stress Check at Cheek Plate Weld
            </h3>
            <div className='Calculator-Side-A' style={{ width: '15%', justifyContent: 'center' }}>
              {props.StressCheckatCheekPlateWeldCondition}
            </div>
          </div>

        </div> */}
      </div >
    </>
  )
}
