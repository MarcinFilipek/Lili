# STM32CubeF0 HAL + CMSIS (vendored in Drivers/, version 1.11.5)
# Odpowiednik tego, co PlatformIO dostarczało przez "framework = stm32cube"

set(MX_Defines_Syms
    USE_HAL_DRIVER
    STM32F042x6
    $<$<CONFIG:Debug>:DEBUG>
)

set(MX_Include_Dirs
    ${CMAKE_SOURCE_DIR}/Core/Inc
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F0xx_HAL_Driver/Inc
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F0xx_HAL_Driver/Inc/Legacy
    ${CMAKE_SOURCE_DIR}/Drivers/CMSIS/Device/ST/STM32F0xx/Include
    ${CMAKE_SOURCE_DIR}/Drivers/CMSIS/Include
)

file(GLOB STM32_Drivers_Src CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F0xx_HAL_Driver/Src/*.c
)

set(MX_Application_Src
    ${CMAKE_SOURCE_DIR}/Core/Src/system_stm32f0xx.c
    ${CMAKE_SOURCE_DIR}/Core/Startup/startup_stm32f042x6.s
)

add_library(STM32_Drivers OBJECT)
target_sources(STM32_Drivers PRIVATE ${STM32_Drivers_Src})
target_compile_definitions(STM32_Drivers PRIVATE ${MX_Defines_Syms})
target_include_directories(STM32_Drivers PRIVATE ${MX_Include_Dirs})

add_library(stm32f0_hal INTERFACE)
target_include_directories(stm32f0_hal INTERFACE ${MX_Include_Dirs})
target_compile_definitions(stm32f0_hal INTERFACE ${MX_Defines_Syms})
target_sources(stm32f0_hal INTERFACE ${MX_Application_Src})
target_link_libraries(stm32f0_hal INTERFACE STM32_Drivers $<TARGET_OBJECTS:STM32_Drivers>)
