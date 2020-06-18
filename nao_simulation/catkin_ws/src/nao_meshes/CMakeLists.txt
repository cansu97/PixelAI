cmake_minimum_required(VERSION 2.8)
project(nao_meshes)

find_package(catkin)
catkin_package()

# When getting a new binay, the MD5 might change, and the folder hierarchy too
# read comments below to adapt

set(TMP_DIR ${CATKIN_DEVEL_PREFIX}/tmp)

set(INSTALLER_TMP_PATH ${TMP_DIR}/tmp/installer.run)

# Try to figure out the architecture
string(FIND ${CATKIN_DEVEL_PREFIX} "i686" IS_32)
message(STATUS "Architecture is apparently 32 (-1 for no): ${IS_32}")
if(${IS_32} STREQUAL "-1")
# Trust CMAKE_SYSTEM_PROCESSOR which does not work on the farm
if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
set(IS_32 FALSE)
else()
set(IS_32 TRUE)
endif()
endif()

if (NOT ${IS_32})
message(STATUS "Downloading 64 bit installer")
file(DOWNLOAD
"https://github.com/ros-naoqi/nao_meshes_installer/raw/master/naomeshes-0.6.7-linux-x64-installer.run"
  ${INSTALLER_TMP_PATH} SHOW_PROGRESS
  INACTIVITY_TIMEOUT 60
  EXPECTED_MD5 5f897a3328f217bae6c0ed2a934e524c
)
else()
message(STATUS "Downloading 32 bit installer")
file(DOWNLOAD
"https://github.com/ros-naoqi/nao_meshes_installer/raw/master/naomeshes-0.6.7-linux-installer.run"
  ${INSTALLER_TMP_PATH} SHOW_PROGRESS
  INACTIVITY_TIMEOUT 60
  EXPECTED_MD5 1984b2790930044b4ee00496315c0021
)
endif()

# specify the interesting folder in the archive. You might want to change
# that for a new binary where the structure is different
set(TMP_MESH_PATH ${TMP_DIR}/meshes)
set(TMP_TEXTURE_PATH ${TMP_DIR}/texture)

# final path in devel/install space
set(FINAL_MESH_PATH ${CATKIN_GLOBAL_SHARE_DESTINATION}/${PROJECT_NAME}/)

# make the file executable to install
file(COPY ${INSTALLER_TMP_PATH}
     FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
     DESTINATION ${TMP_DIR}
)

set(INSTALLER_PATH ${TMP_DIR}/installer.run)

# add a target for devel space
add_custom_target(${PROJECT_NAME}_meshes
  COMMAND ${INSTALLER_PATH} --mode text --prefix ${TMP_DIR}
  COMMAND mkdir -p ${CATKIN_DEVEL_PREFIX}/${FINAL_MESH_PATH}
  COMMAND cp -fr ${TMP_MESH_PATH} ${CATKIN_DEVEL_PREFIX}/${FINAL_MESH_PATH}/
  COMMAND cp -fr ${TMP_TEXTURE_PATH} ${CATKIN_DEVEL_PREFIX}/${FINAL_MESH_PATH}/
  COMMAND rm -fr ${TMP_DIR}/rollback*
)

# uncompress the meshes during install
install(CODE
"
# uncompress the archive
message(STATUS \"using ${INSTALLER_PATH} to decompress to ${TMP_DIR}\")
execute_process(COMMAND ${INSTALLER_PATH} --mode unattended --prefix ${TMP_DIR}
                OUTPUT_VARIABLE OUT
                ERROR_VARIABLE ERROR
                RESULT_VARIABLE RESULT
)
message(STATUS \"intall out: \${OUT}\")
message(STATUS \"install error: \${ERROR}\")
message(STATUS \"install result: \${RESULT}\")

execute_process(COMMAND ls
                WORKING_DIRECTORY ${TMP_DIR}
                OUTPUT_VARIABLE OUT_LS
                ERROR_VARIABLE ERROR_LS
                RESULT_VARIABLE RESULT_LS
)
message(STATUS \"ls out: \${OUT_LS}\")
message(STATUS \"ls error: \${ERROR_LS}\")
message(STATUS \"ls result: \${RESULT_LS}\")

execute_process(COMMAND file ${TMP_DIR}/installer.run
                OUTPUT_VARIABLE OUT_FILE
                ERROR_VARIABLE ERROR_FILE
                RESULT_VARIABLE RESULT_FILE
)
message(STATUS \"file out: \${OUT_FILE}\")
message(STATUS \"file error: \${ERROR_FILE}\")
message(STATUS \"file result: \${RESULT_FILE}\")
"
)

# install the archive
install(DIRECTORY ${TMP_MESH_PATH}
        DESTINATION ${FINAL_MESH_PATH}/
)
install(DIRECTORY ${TMP_TEXTURE_PATH}
        DESTINATION ${FINAL_MESH_PATH}/
)
